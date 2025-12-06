"""
Visualize comparison between baseline and debiased models.
Improved version with value labels, professional colors, and two-column layout.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for professional appearance
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

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

# Professional color palette (suitable for two-column layout)
colors_palette = {
    'random': '#95a5a6',      # Gray (keep for random baseline)
    'hypothesis': '#FFBE7A',  # Light Orange/Peach
    'baseline': '#82B0D2',    # Light Blue/Sky Blue
    'debiased': '#8ECFC9'     # Light Teal/Aqua
}

# Create comparison visualization - optimized for two-column layout
print("Creating comparison charts...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Wider and shorter for two-column

# Figure 1: Overall accuracy comparison
models = ['Random', 'Hypothesis-\nOnly', 'Baseline', 'Debiased']
accuracies = [random_baseline, hyp_acc, baseline_acc, debiased_acc]
colors = [colors_palette['random'], colors_palette['hypothesis'], 
          colors_palette['baseline'], colors_palette['debiased']]

bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
axes[0].axhline(y=random_baseline, color=colors_palette['random'], linestyle='--', alpha=0.4, linewidth=1.5)
axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[0].set_title('Overall Model Performance', fontsize=12, fontweight='bold', pad=10)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

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
bars2_base = axes[1].bar(x - width/2, baseline_class_accs, width, label='Baseline', 
                         alpha=0.85, color=colors_palette['baseline'], edgecolor='white', linewidth=1.5)
bars2_deb = axes[1].bar(x + width/2, debiased_class_accs, width, label='Debiased', 
                        alpha=0.85, color=colors_palette['debiased'], edgecolor='white', linewidth=1.5)

axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[1].set_title('Per-Class Accuracy Comparison', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(classes, fontsize=10)
axes[1].legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add value labels on bars for per-class comparison
for bars, accs in [(bars2_base, baseline_class_accs), (bars2_deb, debiased_class_accs)]:
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout(pad=2.0)
output_dir = os.path.join(project_root, 'outputs', 'evaluations')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'baseline_vs_debiased_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Comparison chart saved to: {output_path}")
plt.close()

print("Comparison visualizations completed!")

