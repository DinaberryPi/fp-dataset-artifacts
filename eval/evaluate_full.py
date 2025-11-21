"""
Evaluate the debiased model on the FULL validation set (not just 500 examples)
to get a fair comparison with the baseline.
"""

import argparse
import json
import os
import sys

# Disable external logging integrations (e.g., Weights & Biases) by default
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import datasets
from datasets import logging as ds_logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import logging as hf_logging

# Add parent directory to path for helpers import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.helpers import prepare_dataset_nli, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the debiased model on the full validation set.")
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Suppress console logs and progress bars.')
    parser.add_argument('--model-path', default='../outputs/evaluations/debiased_model/',
                        help='Path to the model checkpoint to evaluate.')
    parser.add_argument('--output-dir', default='../outputs/evaluations/debiased_model_full_eval/',
                        help='Directory to store evaluation artifacts.')
    return parser.parse_args()


def main():
    args = parse_args()
    quiet = args.quiet
    log = (lambda *a, **k: None) if quiet else print

    if quiet:
        hf_logging.set_verbosity_error()
        ds_logging.set_verbosity_error()
        ds_logging.disable_progress_bar()
    else:
        hf_logging.set_verbosity_warning()
        ds_logging.set_verbosity_warning()

    model_path = args.model_path
    output_dir = args.output_dir
    
    log("=" * 80)
    log("EVALUATING DEBIASED MODEL ON FULL VALIDATION SET")
    log("This will give us a fair comparison with baseline!")
    log("=" * 80)
    
    # Load dataset
    log("\nLoading SNLI dataset...")
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Load debiased model
    log(f"\nLoading debiased model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Prepare FULL validation set (no max_eval_samples!)
    log("\nPreparing FULL validation dataset...")
    eval_dataset = dataset['validation']
    log(f"Total validation examples: {len(eval_dataset)}")
    
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
        report_to=[],
        disable_tqdm=quiet,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )
    
    log("\n" + "=" * 80)
    log("EVALUATING ON FULL VALIDATION SET...")
    log("=" * 80 + "\n")
    
    # Evaluate
    results = trainer.evaluate()
    
    log("\n" + "=" * 80)
    log("FULL VALIDATION RESULTS:")
    log(f"Debiased Model Accuracy: {results['eval_accuracy']:.2%}")
    log("\nComparison:")
    log(f"  Baseline (full validation):  76.09%")
    log(f"  Debiased (full validation):  {results['eval_accuracy']:.2%}")
    log(f"  Improvement:                 {(results['eval_accuracy'] - 0.7609):+.2%}")
    log("=" * 80)
    
    # Save predictions
    log("\nGenerating predictions for full validation set...")
    predictions_output = trainer.predict(eval_dataset_featurized)
    
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, 'eval_predictions_full.jsonl')
    
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset['validation']):
            pred_dict = dict(example)
            pred_dict['predicted_scores'] = predictions_output.predictions[i].tolist()
            pred_dict['predicted_label'] = int(predictions_output.predictions[i].argmax())
            f.write(json.dumps(pred_dict) + '\n')
    
    log(f"Predictions saved to: {predictions_path}")
    log("\nNow you have a fair comparison!")
    log("Both models evaluated on the SAME full validation set.")

if __name__ == "__main__":
    main()


