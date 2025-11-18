"""
Visualize baseline model results: confusion matrix and per-class accuracy.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the project root directory (parent of analyze/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Load baseline predictions
print("Loading baseline predictions...")
baseline_predictions = []
predictions_path = os.path.join(project_root, 'outputs', 'evaluations', 'baseline_100k', 'eval_predictions.jsonl')
with open(predictions_path, 'r', encoding='utf-8') as f:
    for line in f:
        baseline_predictions.append(json.loads(line))

label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Create confusion matrix
print("Creating confusion matrix...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

baseline_confusion = np.zeros((3, 3))
for p in baseline_predictions:
    baseline_confusion[p['label']][p['predicted_label']] += 1

# Normalize
baseline_confusion_norm = baseline_confusion / baseline_confusion.sum(axis=1, keepdims=True)

sns.heatmap(baseline_confusion_norm, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Entail', 'Neutral', 'Contrad'],
            yticklabels=['Entail', 'Neutral', 'Contrad'],
            ax=ax, cbar_kws={'label': 'Proportion'})
ax.set_title('Baseline Model Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)

plt.tight_layout()
output_dir = os.path.join(project_root, 'outputs', 'evaluations')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'baseline_confusion_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {output_path}")
plt.close()

# Per-class accuracy visualization
print("Creating per-class accuracy chart...")
baseline_class_accs = []
for label in [0, 1, 2]:
    baseline_class = [p for p in baseline_predictions if p['label'] == label]
    baseline_class_acc = sum(1 for p in baseline_class if p['predicted_label'] == label) / len(baseline_class)
    baseline_class_accs.append(baseline_class_acc)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
classes = ['Entailment', 'Neutral', 'Contradiction']
colors = ['#2ecc71', '#f39c12', '#e74c3c']

bars = ax.bar(classes, baseline_class_accs, color=colors, alpha=0.7)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Baseline Model - Per-Class Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, baseline_class_accs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(output_dir, 'baseline_per_class_accuracy.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Per-class accuracy chart saved to: {output_path}")
plt.close()

print("Baseline visualizations completed!")

