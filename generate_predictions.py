"""
Generate predictions from a trained model without retraining.
Useful when you forgot to save predictions during training.
"""

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from helpers import prepare_dataset_nli_hypothesis_only, compute_accuracy
import os
import json

def generate_predictions(model_path, output_file, hypothesis_only=False):
    """
    Generate predictions from a saved model.
    
    Args:
        model_path: Path to saved model (e.g., './hypothesis_only_model/')
        output_file: Where to save predictions (e.g., 'hypothesis_only_model/eval_predictions.jsonl')
        hypothesis_only: If True, use hypothesis-only preprocessing
    """
    print(f"Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load dataset
    print("Loading SNLI validation set...")
    dataset = datasets.load_dataset('snli')
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    eval_dataset = dataset['validation']
    
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Preprocess
    print("Preprocessing...")
    if hypothesis_only:
        prepare_fn = lambda exs: prepare_dataset_nli_hypothesis_only(exs, tokenizer, 128)
    else:
        from helpers import prepare_dataset_nli
        prepare_fn = lambda exs: prepare_dataset_nli(exs, tokenizer, 128)
    
    eval_dataset_featurized = eval_dataset.map(
        prepare_fn,
        batched=True,
        num_proc=2,
        remove_columns=eval_dataset.column_names
    )
    
    # Create trainer for prediction
    training_args = TrainingArguments(
        output_dir=model_path,
        per_device_eval_batch_size=32,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy
    )
    
    # Generate predictions
    print("Generating predictions...")
    predictions_output = trainer.predict(eval_dataset_featurized)
    
    # Calculate accuracy
    accuracy = compute_accuracy(predictions_output)
    print(f"\nAccuracy: {accuracy['accuracy']:.2%}")
    
    # Save to file
    print(f"Saving predictions to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(eval_dataset):
            example_with_prediction = dict(example)
            example_with_prediction['predicted_scores'] = predictions_output.predictions[i].tolist()
            example_with_prediction['predicted_label'] = int(predictions_output.predictions[i].argmax())
            f.write(json.dumps(example_with_prediction))
            f.write('\n')
    
    print(f"✅ Done! Saved {len(eval_dataset)} predictions")
    return accuracy['accuracy']

if __name__ == "__main__":
    # Example usage - generate predictions for hypothesis-only model
    print("=" * 80)
    print("GENERATING PREDICTIONS FROM SAVED MODEL")
    print("=" * 80)
    
    generate_predictions(
        model_path='./hypothesis_only_model/',
        output_file='./hypothesis_only_model/eval_predictions.jsonl',
        hypothesis_only=True
    )

