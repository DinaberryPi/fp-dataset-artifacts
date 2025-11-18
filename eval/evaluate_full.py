"""
Evaluate the debiased model on the FULL validation set (not just 500 examples)
to get a fair comparison with the baseline.
"""

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import os
import sys

# Add parent directory to path for helpers import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.helpers import prepare_dataset_nli, compute_accuracy

def main():
    model_path = '../outputs/evaluations/debiased_model/'
    output_dir = '../outputs/evaluations/debiased_model_full_eval/'
    
    print("=" * 80)
    print("EVALUATING DEBIASED MODEL ON FULL VALIDATION SET")
    print("This will give us a fair comparison with baseline!")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading SNLI dataset...")
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Load debiased model
    print(f"\nLoading debiased model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare FULL validation set (no max_eval_samples!)
    print("\nPreparing FULL validation dataset...")
    eval_dataset = dataset['validation']
    print(f"Total validation examples: {len(eval_dataset)}")
    
    prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=2,
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments (for evaluation only)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=32,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )
    
    print("\n" + "=" * 80)
    print("EVALUATING ON FULL VALIDATION SET...")
    print("=" * 80 + "\n")
    
    # Evaluate
    results = trainer.evaluate()
    
    print("\n" + "=" * 80)
    print("FULL VALIDATION RESULTS:")
    print(f"Debiased Model Accuracy: {results['eval_accuracy']:.2%}")
    print("\nComparison:")
    print(f"  Baseline (full validation):  76.09%")
    print(f"  Debiased (full validation):  {results['eval_accuracy']:.2%}")
    print(f"  Improvement:                 {(results['eval_accuracy'] - 0.7609):+.2%}")
    print("=" * 80)
    
    # Save predictions
    print("\nGenerating predictions for full validation set...")
    predictions_output = trainer.predict(eval_dataset_featurized)
    
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, 'eval_predictions_full.jsonl')
    
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset['validation']):
            pred_dict = dict(example)
            pred_dict['predicted_scores'] = predictions_output.predictions[i].tolist()
            pred_dict['predicted_label'] = int(predictions_output.predictions[i].argmax())
            f.write(json.dumps(pred_dict) + '\n')
    
    print(f"Predictions saved to: {predictions_path}")
    print("\nNow you have a fair comparison!")
    print("Both models evaluated on the SAME full validation set.")

if __name__ == "__main__":
    main()


