"""
Verify all three prediction files from Phase 2 training.
"""
import json

def verify_predictions(filepath, model_name):
    """Check if predictions file is valid"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines
        data_lines = [l for l in lines if l.strip()]
        num_examples = len(data_lines)
        
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        print(f"[OK] File found: {filepath}")
        print(f"[OK] Number of examples: {num_examples}")
        
        # Expected: 9842 for full validation set
        if num_examples == 9842:
            print("[OK] PERFECT! Full validation set (9,842 examples)")
        else:
            print(f"[WARNING] Expected 9,842 examples, got {num_examples}")
        
        # Check first example format
        first = json.loads(data_lines[0])
        required_keys = ['premise', 'hypothesis', 'label', 'predicted_label', 'predicted_scores']
        missing = [k for k in required_keys if k not in first]
        
        if not missing:
            print("[OK] All required fields present")
        else:
            print(f"[ERROR] Missing fields: {missing}")
        
        # Calculate accuracy
        correct = 0
        for line in data_lines:
            example = json.loads(line)
            if example['label'] == example['predicted_label']:
                correct += 1
        
        accuracy = (correct / num_examples) * 100
        print(f"[OK] Accuracy: {accuracy:.2f}%")
        
        # Per-class breakdown
        class_names = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}
        class_counts = {0: 0, 1: 0, 2: 0}
        class_correct = {0: 0, 1: 0, 2: 0}
        
        for line in data_lines:
            example = json.loads(line)
            true_label = example['label']
            pred_label = example['predicted_label']
            class_counts[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1
        
        print(f"\nPer-Class Accuracy:")
        for label_id in [0, 1, 2]:
            class_acc = (class_correct[label_id] / class_counts[label_id]) * 100
            print(f"   {class_names[label_id]:15s}: {class_acc:.2f}% ({class_correct[label_id]}/{class_counts[label_id]})")
        
        return accuracy
        
    except FileNotFoundError:
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        print(f"[ERROR] File not found: {filepath}")
        return None
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        print(f"[ERROR] Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 2: 100K TRAINING - VERIFICATION")
    print("="*70)
    
    # Verify all three models
    files_to_check = [
        ('../outputs/evaluations/baseline_100k/eval_predictions.jsonl', 'Baseline Model (100K)'),
        ('../outputs/evaluations/hypothesis_only_model/eval_predictions.jsonl', 'Hypothesis-Only Model (Artifact Detector)'),
        ('../outputs/evaluations/debiased_model/eval_predictions.jsonl', 'Debiased Model (100K)')
    ]
    
    accuracies = {}
    
    for filepath, model_name in files_to_check:
        accuracy = verify_predictions(filepath, model_name)
        if accuracy:
            accuracies[model_name] = accuracy
    
    # Summary
    if len(accuracies) == 3:
        print(f"\n{'='*70}")
        print("SUMMARY - ALL MODELS")
        print('='*70)
        print(f"Random Baseline:           33.33% (guessing)")
        print(f"Hypothesis-Only:           {accuracies['Hypothesis-Only Model (Artifact Detector)']:.2f}% (artifacts detected!)")
        print(f"Baseline Model:            {accuracies['Baseline Model (100K)']:.2f}%")
        print(f"Debiased Model:            {accuracies['Debiased Model (100K)']:.2f}%")
        
        improvement = accuracies['Debiased Model (100K)'] - accuracies['Baseline Model (100K)']
        print(f"\nDebiasing Effect:        {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"[SUCCESS] Debiasing improved performance!")
        elif improvement > -0.5:
            print(f"[NEUTRAL] Minimal change (within noise)")
        else:
            print(f"[INFO] Lower accuracy, but may be more robust on hard examples")
        
        artifact_strength = accuracies['Hypothesis-Only Model (Artifact Detector)'] - 33.33
        print(f"\nArtifact Strength:       +{artifact_strength:.2f}% above random")
        print(f"   (Higher = stronger artifacts in dataset)")
        
        print("\n" + "="*70)
        print("[OK] All files verified! Ready for analysis.")
        print("="*70)
    else:
        print(f"\n[WARNING] Warning: Only {len(accuracies)}/3 files found")

