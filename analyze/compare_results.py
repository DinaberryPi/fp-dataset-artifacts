"""
Compare baseline and debiased model results, including per-class accuracy and prediction changes.
"""
import json
import os

# Get the project root directory (parent of analyze/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print("=" * 80)
print("Results Comparison - Baseline vs Debiased")
print("=" * 80)

# Load metrics
with open(os.path.join(project_root, 'outputs', 'evaluations', 'baseline_100k', 'eval_metrics.json'), 'r') as f:
    baseline_metrics = json.load(f)

with open(os.path.join(project_root, 'outputs', 'evaluations', 'hypothesis_only_model', 'eval_metrics.json'), 'r') as f:
    hyp_metrics = json.load(f)

with open(os.path.join(project_root, 'outputs', 'evaluations', 'debiased_model', 'eval_metrics.json'), 'r') as f:
    debiased_metrics = json.load(f)

# Load predictions
baseline_predictions = []
with open(os.path.join(project_root, 'outputs', 'evaluations', 'baseline_100k', 'eval_predictions.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        baseline_predictions.append(json.loads(line))

debiased_predictions = []
with open(os.path.join(project_root, 'outputs', 'evaluations', 'debiased_model', 'eval_predictions.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        debiased_predictions.append(json.loads(line))

label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Calculate statistics
random_baseline = 1.0 / 3.0
baseline_acc = baseline_metrics['eval_accuracy']
hyp_acc = hyp_metrics['eval_accuracy']
debiased_acc = debiased_metrics['eval_accuracy']

print(f"\nRandom Baseline:        {random_baseline:.4f} ({random_baseline*100:.2f}%)")
print(f"Hypothesis-Only:        {hyp_acc:.4f} ({hyp_acc*100:.2f}%) [Above random: +{(hyp_acc-random_baseline)*100:.2f}%]")
print(f"Baseline (Full Model):  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Debiased:               {debiased_acc:.4f} ({debiased_acc*100:.2f}%) [Change: {(debiased_acc-baseline_acc)*100:+.2f}%]")

print("\n" + "=" * 80)
print("Key Findings:")
print("=" * 80)
print(f"1. Hypothesis-Only model achieves {hyp_acc*100:.2f}%, proving strong artifacts exist!")
print(f"2. Debiasing maintains performance: {debiased_acc*100:.2f}% vs {baseline_acc*100:.2f}%")
print(f"3. {'Debiasing preserved performance' if abs(debiased_acc - baseline_acc) < 0.01 else 'Debiasing affected performance'}")

# Calculate per-class accuracy comparison
print("\n" + "=" * 80)
print("Per-Class Accuracy Comparison")
print("=" * 80)

for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_predictions if p['label'] == label]
    debiased_class = [p for p in debiased_predictions if p['label'] == label]
    
    baseline_class_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    debiased_class_acc = sum(1 for p in debiased_class if p['predicted_label'] == label) / len(debiased_class)
    
    change = debiased_class_acc - baseline_class_acc
    print(f"{label_names[label]:15}: Baseline={baseline_class_acc:.2%}, Debiased={debiased_class_acc:.2%}, Change={change:+.2%}")

# Prediction changes
changes = []
for i, (base, deb) in enumerate(zip(baseline_predictions, debiased_predictions)):
    if base['predicted_label'] != deb['predicted_label']:
        changes.append({
            'index': i,
            'premise': base['premise'],
            'hypothesis': base['hypothesis'],
            'true_label': base['label'],
            'baseline_pred': base['predicted_label'],
            'debiased_pred': deb['predicted_label'],
        })

baseline_wrong_debiased_right = [c for c in changes if c['baseline_pred'] != c['true_label'] and c['debiased_pred'] == c['true_label']]
baseline_right_debiased_wrong = [c for c in changes if c['baseline_pred'] == c['true_label'] and c['debiased_pred'] != c['true_label']]

print(f"\n" + "=" * 80)
print("Prediction Changes")
print("=" * 80)
print(f"Total predictions changed: {len(changes)} ({len(changes)/len(baseline_predictions):.1%})")
print(f"Baseline wrong -> Debiased correct (FIXES): {len(baseline_wrong_debiased_right)}")
print(f"Baseline correct -> Debiased wrong (BREAKS): {len(baseline_right_debiased_wrong)}")
print(f"Net improvement: {len(baseline_wrong_debiased_right) - len(baseline_right_debiased_wrong):+d}")

# Save fixes for later use
fixes_file = os.path.join(project_root, 'outputs', 'evaluations', 'fixes_examples.json')
os.makedirs(os.path.dirname(fixes_file), exist_ok=True)
with open(fixes_file, 'w', encoding='utf-8') as f:
    json.dump(baseline_wrong_debiased_right[:10], f, indent=2, ensure_ascii=False)

print(f"\nTop 10 fixes saved to: {fixes_file}")

