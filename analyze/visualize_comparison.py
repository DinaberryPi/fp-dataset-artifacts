"""
Visualize comparison between baseline and debiased models.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the project root directory (parent of analyze/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load metrics
print("Loading metrics...")
with open(os.path.join(project_root, 'outputs', 'evaluations', 'baseline_100k', 'eval_metrics.json'), 'r') as f:
    baseline_metrics = json.load(f)

with open(os.path.join(project_root, 'outputs', 'evaluations', 'hypothesis_only_model', 'eval_metrics.json'), 'r') as f:
    hyp_metrics = json.load(f)

with open(os.path.join(project_root, 'outputs', 'evaluations', 'debiased_model', 'eval_metrics.json'), 'r') as f:
    debiased_metrics = json.load(f)

# Load predictions
print("Loading predictions...")
baseline_predictions = []
with open(os.path.join(project_root, 'outputs', 'evaluations', 'baseline_100k', 'eval_predictions.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        baseline_predictions.append(json.loads(line))

debiased_predictions = []
with open(os.path.join(project_root, 'outputs', 'evaluations', 'debiased_model', 'eval_predictions.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        debiased_predictions.append(json.loads(line))

# Calculate statistics
random_baseline = 1.0 / 3.0
baseline_acc = baseline_metrics['eval_accuracy']
hyp_acc = hyp_metrics['eval_accuracy']
debiased_acc = debiased_metrics['eval_accuracy']

# Create comparison visualization
print("Creating comparison charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Figure 1: Overall accuracy comparison
models = ['Random', 'Hypothesis-\nOnly', 'Baseline', 'Debiased']
accuracies = [random_baseline, hyp_acc, baseline_acc, debiased_acc]
colors = ['gray', 'orange', 'blue', 'green']

axes[0].bar(models, accuracies, color=colors, alpha=0.7)
axes[0].axhline(y=random_baseline, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Overall Model Performance', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[0].text(i, acc + 0.02, f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

# Figure 2: Per-class accuracy comparison
classes = ['Entailment', 'Neutral', 'Contradiction']
baseline_class_accs = []
debiased_class_accs = []

for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_predictions if p['label'] == label]
    debiased_class = [p for p in debiased_predictions if p['label'] == label]
    
    baseline_class_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    debiased_class_acc = sum(1 for p in debiased_class if p['predicted_label'] == label) / len(debiased_class)
    
    baseline_class_accs.append(baseline_class_acc)
    debiased_class_accs.append(debiased_class_acc)

x = np.arange(len(classes))
width = 0.35
axes[1].bar(x - width/2, baseline_class_accs, width, label='Baseline', alpha=0.7, color='blue')
axes[1].bar(x + width/2, debiased_class_accs, width, label='Debiased', alpha=0.7, color='green')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes)
axes[1].legend(fontsize=11)
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
output_dir = os.path.join(project_root, 'outputs', 'evaluations')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'baseline_vs_debiased_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparison chart saved to: {output_path}")
plt.close()

print("Comparison visualizations completed!")

