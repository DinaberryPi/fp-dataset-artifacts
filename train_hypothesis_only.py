"""
Train a hypothesis-only model to capture dataset artifacts.
This model will be used for debiasing in the main model.
"""

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from helpers import prepare_dataset_nli_hypothesis_only, compute_accuracy
import os

def main():
    # Training configuration
    output_dir = './hypothesis_only_model/'
    model_name = 'google/electra-small-discriminator'
    max_train_samples = 100000  # Phase 2: 100K training
    max_eval_samples = None      # Phase 2: Full validation (9,842)
    num_train_epochs = 3
    per_device_train_batch_size = 16
    
    print("=" * 80)
    print("TRAINING HYPOTHESIS-ONLY MODEL (Artifact Model)")
    print("This model only sees the hypothesis, not the premise!")
    print("It will learn to exploit dataset artifacts.")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading SNLI dataset...")
    dataset = datasets.load_dataset('snli')
    
    # Remove examples with no label
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Initialize model and tokenizer
    print(f"\nLoading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # Make tensor contiguous if needed
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Prepare datasets
    print("\nPreparing datasets (hypothesis-only)...")
    train_dataset = dataset['train']
    if max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    
    eval_dataset = dataset['validation']
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    # Preprocess: ONLY use hypothesis!
    prepare_fn = lambda exs: prepare_dataset_nli_hypothesis_only(exs, tokenizer, 128)
    
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
        eval_strategy="epoch",  # Fixed: was evaluation_strategy
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("Remember: This model CANNOT see the premise!")
    print("High accuracy = strong artifacts in the dataset")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # Evaluate and get predictions
    print("\n" + "=" * 80)
    print("EVALUATING HYPOTHESIS-ONLY MODEL...")
    print("=" * 80)
    
    # Get predictions
    predictions_output = trainer.predict(eval_dataset_featurized)
    results = trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS-ONLY MODEL RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
    print(f"Loss:     {results['eval_loss']:.4f}")
    print("=" * 80)
    
    # Save the model
    trainer.save_model()
    print(f"\nModel saved to: {output_dir}")
    
    # Save evaluation metrics to JSON
    import json
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'eval_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {os.path.join(output_dir, 'eval_metrics.json')}")
    
    # Save predictions to jsonl file
    print("\nSaving predictions...")
    
    with open(os.path.join(output_dir, 'eval_predictions.jsonl'), 'w', encoding='utf-8') as f:
        for i, example in enumerate(eval_dataset):
            example_with_prediction = dict(example)
            example_with_prediction['predicted_scores'] = predictions_output.predictions[i].tolist()
            example_with_prediction['predicted_label'] = int(predictions_output.predictions[i].argmax())
            f.write(json.dumps(example_with_prediction))
            f.write('\n')
    
    print(f"Predictions saved to: {os.path.join(output_dir, 'eval_predictions.jsonl')}")
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()

