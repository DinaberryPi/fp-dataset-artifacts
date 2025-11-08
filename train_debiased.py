"""
Train a debiased model using the hypothesis-only model to identify artifacts.
Uses example reweighting: downweight examples where hypothesis-only model is confident.
"""

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from helpers import prepare_dataset_nli, compute_accuracy
import torch
import numpy as np
from typing import Dict
import os

class DebiasedTrainer(Trainer):
    """
    Custom Trainer that reweights examples based on hypothesis-only model confidence.
    Examples where the bias model is confident get lower weight.
    """
    
    def __init__(self, *args, bias_model=None, bias_tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_model = bias_model
        self.bias_tokenizer = bias_tokenizer
        
        # Move bias model to same device as main model
        if self.bias_model is not None:
            self.bias_model.to(self.args.device)
            self.bias_model.eval()  # Keep in eval mode
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss based on bias model confidence.
        """
        labels = inputs.get("labels")
        
        # Get main model outputs
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Get bias model predictions if available
        weights = None
        if self.bias_model is not None:
            with torch.no_grad():
                # Create a copy of inputs to avoid any gradient tracking issues
                bias_inputs = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                              for k, v in inputs.items()}
                
                bias_outputs = self.bias_model(**bias_inputs)
                bias_logits = bias_outputs.logits
                bias_probs = torch.nn.functional.softmax(bias_logits, dim=-1)
                
                # Get confidence: max probability
                bias_confidence = bias_probs.max(dim=-1)[0]
                
                # Compute weights: lower weight for high confidence examples
                # If bias model is confident, it's likely using artifacts
                # Formula: weight = 1 / (1 + bias_confidence)
                # This gives weight ≈ 0.5 when bias_confidence is high
                # and weight ≈ 1.0 when bias_confidence is low
                weights = 1.0 / (1.0 + bias_confidence)
                
                # Convert to numpy and back to ensure no gradient connection
                weights = torch.tensor(weights.cpu().numpy(), 
                                     device=logits.device, 
                                     dtype=logits.dtype,
                                     requires_grad=False)
        
        # Compute standard cross-entropy loss (no reduction yet)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_example = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Apply weights if computed
        if weights is not None:
            loss_per_example = loss_per_example * weights
        
        # Average the loss
        loss = loss_per_example.mean()
        
        return (loss, outputs) if return_outputs else loss


def main():
    # Configuration
    output_dir = './debiased_model/'
    bias_model_path = './hypothesis_only_model/'  # The artifact model we just trained
    model_name = 'google/electra-small-discriminator'
    max_train_samples = 100000  # Phase 2: 100K training
    max_eval_samples = None      # Phase 2: Full validation (9,842)
    num_train_epochs = 3
    per_device_train_batch_size = 16
    
    print("=" * 80)
    print("TRAINING DEBIASED MODEL")
    print("Using hypothesis-only model to identify and downweight artifacts")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading SNLI dataset...")
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Load main model
    print(f"\nLoading main model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Load bias model (hypothesis-only)
    print(f"\nLoading bias model from: {bias_model_path}")
    bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
    bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_path)
    
    print("✓ Bias model loaded successfully!")
    print("  This model will identify artifact-based examples during training.")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = dataset['train']
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    
    eval_dataset = dataset['validation']
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    # Preprocess: Use BOTH premise and hypothesis
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)
    
    train_dataset_featurized = train_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=2,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=2,
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Initialize debiased trainer
    trainer = DebiasedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
        bias_model=bias_model,  # Pass the artifact model
        bias_tokenizer=bias_tokenizer
    )
    
    print("\n" + "=" * 80)
    print("STARTING DEBIASED TRAINING...")
    print("Strategy: Downweight examples where bias model is confident")
    print("=" * 80 + "\n")
    
    # Train
    trainer.train()
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING DEBIASED MODEL...")
    print("=" * 80)
    results = trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("DEBIASED MODEL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
    print(f"Loss:     {results['eval_loss']:.4f}")
    print("=" * 80)
    
    # Save model
    trainer.save_model()
    print(f"\nDebiased model saved to: {output_dir}")
    
    # Save evaluation metrics to JSON
    import json
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {os.path.join(output_dir, 'eval_metrics.json')}")
    
    # Save predictions for analysis
    print("\nSaving predictions for comparison...")
    predictions_output = trainer.predict(eval_dataset_featurized)
    
    # Get the eval dataset (handle None for max_eval_samples)
    eval_examples = dataset['validation']
    if max_eval_samples:
        eval_examples = eval_examples.select(range(max_eval_samples))
    
    with open(os.path.join(output_dir, 'eval_predictions.jsonl'), 'w', encoding='utf-8') as f:
        for i, example in enumerate(eval_examples):
            pred_dict = dict(example)
            pred_dict['predicted_scores'] = predictions_output.predictions[i].tolist()
            pred_dict['predicted_label'] = int(predictions_output.predictions[i].argmax())
            f.write(json.dumps(pred_dict) + '\n')
    
    print(f"Predictions saved to: {output_dir}/eval_predictions.jsonl")
    print(f"Total predictions: {len(eval_examples)}")
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()

