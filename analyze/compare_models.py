"""
Compare baseline and debiased model predictions.
"""

import json
from collections import Counter

print("=" * 80)
print("COMPARING BASELINE vs DEBIASED MODEL")
print("=" * 80)

# Load predictions
print("\nLoading predictions...")
# Phase 2: Use 100K predictions
baseline_path = '../outputs/evaluations/baseline_100k/eval_predictions.jsonl'
debiased_path = '../outputs/evaluations/debiased_model/eval_predictions.jsonl'

baseline_preds = []
print(f"Loading baseline from: {baseline_path}")
with open(baseline_path, 'r', encoding='utf-8') as f:
    for line in f:
        baseline_preds.append(json.loads(line))

debiased_preds = []
print(f"Loading debiased from: {debiased_path}")
with open(debiased_path, 'r', encoding='utf-8') as f:
    for line in f:
        debiased_preds.append(json.loads(line))

# Check if same number of examples
print(f"\nBaseline predictions: {len(baseline_preds)}")
print(f"Debiased predictions: {len(debiased_preds)}")

if len(baseline_preds) != len(debiased_preds):
    print(f"\nWARNING: Different number of predictions!")
    print(f"Using first {min(len(baseline_preds), len(debiased_preds))} examples for comparison")
    # Truncate to same length
    min_len = min(len(baseline_preds), len(debiased_preds))
    baseline_preds = baseline_preds[:min_len]
    debiased_preds = debiased_preds[:min_len]

label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Overall accuracy
baseline_correct = sum(1 for p in baseline_preds if p['label'] == p['predicted_label'])
debiased_correct = sum(1 for p in debiased_preds if p['label'] == p['predicted_label'])

print(f"\n=== OVERALL ACCURACY ===")
print(f"Baseline: {baseline_correct/len(baseline_preds):.2%} ({baseline_correct}/{len(baseline_preds)})")
print(f"Debiased: {debiased_correct/len(debiased_preds):.2%} ({debiased_correct}/{len(debiased_preds)})")
print(f"Change:   {(debiased_correct - baseline_correct)/len(baseline_preds):+.2%}")

# Per-class accuracy
print(f"\n=== PER-CLASS ACCURACY ===")
for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_preds if p['label'] == label]
    debiased_class = [p for p in debiased_preds if p['label'] == label]
    
    baseline_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    debiased_acc = sum(1 for p in debiased_class if p['predicted_label'] == label) / len(debiased_class)
    
    print(f"{label_names[label]:15}: Baseline={baseline_acc:.2%}, Debiased={debiased_acc:.2%}, Change={debiased_acc-baseline_acc:+.2%}")

# Check negation words
print(f"\n=== NEGATION WORD ANALYSIS ===")
negation_words = ['no', 'not', 'never', 'nobody', 'nothing', 'nowhere', 'neither', 'none', "n't"]

def has_negation(text):
    text_lower = text.lower()
    return any(neg in text_lower for neg in negation_words)

# Find examples with negation
neg_indices = [i for i, p in enumerate(baseline_preds) if has_negation(p['hypothesis'])]
print(f"Examples with negation: {len(neg_indices)}")

baseline_neg_correct = sum(1 for i in neg_indices if baseline_preds[i]['label'] == baseline_preds[i]['predicted_label'])
debiased_neg_correct = sum(1 for i in neg_indices if debiased_preds[i]['label'] == debiased_preds[i]['predicted_label'])

print(f"Baseline accuracy on negation examples: {baseline_neg_correct/len(neg_indices):.2%}")
print(f"Debiased accuracy on negation examples: {debiased_neg_correct/len(neg_indices):.2%}")
print(f"Improvement: {(debiased_neg_correct - baseline_neg_correct)/len(neg_indices):+.2%}")

# Predictions that changed
print(f"\n=== PREDICTION CHANGES ===")
changes = []
for i, (base, deb) in enumerate(zip(baseline_preds, debiased_preds)):
    if base['predicted_label'] != deb['predicted_label']:
        changes.append({
            'index': i,
            'premise': base['premise'],
            'hypothesis': base['hypothesis'],
            'true_label': base['label'],
            'baseline_pred': base['predicted_label'],
            'debiased_pred': deb['predicted_label'],
            'has_negation': has_negation(base['hypothesis'])
        })

print(f"Total predictions that changed: {len(changes)} ({len(changes)/len(baseline_preds):.1%})")

# Categorize changes
baseline_wrong_debiased_right = [c for c in changes if c['baseline_pred'] != c['true_label'] and c['debiased_pred'] == c['true_label']]
baseline_right_debiased_wrong = [c for c in changes if c['baseline_pred'] == c['true_label'] and c['debiased_pred'] != c['true_label']]

print(f"Baseline wrong -> Debiased correct (FIXES): {len(baseline_wrong_debiased_right)}")
print(f"Baseline correct -> Debiased wrong (BREAKS): {len(baseline_right_debiased_wrong)}")
print(f"Net improvement: {len(baseline_wrong_debiased_right) - len(baseline_right_debiased_wrong):+d}")

# Show examples of fixes
print(f"\n=== EXAMPLES OF FIXES (Baseline wrong -> Debiased correct) ===")
for i, ex in enumerate(baseline_wrong_debiased_right[:3], 1):
    print(f"\nFix {i}:")
    print(f"  Premise: {ex['premise']}")
    print(f"  Hypothesis: {ex['hypothesis']}")
    print(f"  Has negation: {ex['has_negation']}")
    print(f"  True label: {label_names[ex['true_label']]}")
    print(f"  Baseline predicted: {label_names[ex['baseline_pred']]} (WRONG)")
    print(f"  Debiased predicted: {label_names[ex['debiased_pred']]} (CORRECT)")

print("\n" + "=" * 80)
print("Comparison complete!")
print("=" * 80)

