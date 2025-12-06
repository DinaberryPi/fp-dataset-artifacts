"""
Visualize negation analysis with professional color scheme.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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

# Professional color palette (using the three-color scheme)
colors_palette = {
    'baseline': '#82B0D2',    # Light Blue/Sky Blue
    'debiased': '#8ECFC9',    # Light Teal/Aqua
    'true': '#FA7F6F',        # Coral/Salmon Pink (red tone for true labels)
    'coral': '#FA7F6F'        # Coral/Salmon Pink
}

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

label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Define negation words
negation_words = ['no', 'not', 'never', 'nobody', 'nothing', 'nowhere', 'neither', 'none', "n't", 'nor']

def has_negation(text):
    """Check if text contains negation words."""
    text_lower = text.lower()
    return any(neg in text_lower for neg in negation_words)

# Analyze negation for baseline
baseline_with_neg = [p for p in baseline_predictions if has_negation(p['hypothesis'])]
baseline_without_neg = [p for p in baseline_predictions if not has_negation(p['hypothesis'])]

# Analyze negation for debiased
debiased_with_neg = [p for p in debiased_predictions if has_negation(p['hypothesis'])]
debiased_without_neg = [p for p in debiased_predictions if not has_negation(p['hypothesis'])]

# Calculate accuracy on negation examples
baseline_neg_correct = sum(1 for p in baseline_with_neg if p['predicted_label'] == p['label'])
baseline_neg_acc = baseline_neg_correct / len(baseline_with_neg) if baseline_with_neg else 0

baseline_no_neg_correct = sum(1 for p in baseline_without_neg if p['predicted_label'] == p['label'])
baseline_no_neg_acc = baseline_no_neg_correct / len(baseline_without_neg) if baseline_without_neg else 0

debiased_neg_correct = sum(1 for p in debiased_with_neg if p['predicted_label'] == p['label'])
debiased_neg_acc = debiased_neg_correct / len(debiased_with_neg) if debiased_with_neg else 0

debiased_no_neg_correct = sum(1 for p in debiased_without_neg if p['predicted_label'] == p['label'])
debiased_no_neg_acc = debiased_no_neg_correct / len(debiased_without_neg) if debiased_without_neg else 0

# Label distribution for negation examples
neg_true_labels = Counter(p['label'] for p in baseline_with_neg)
baseline_neg_preds = Counter(p['predicted_label'] for p in baseline_with_neg)
debiased_neg_preds = Counter(p['predicted_label'] for p in debiased_with_neg)

# Contradiction prediction rates
true_contrad_pct = neg_true_labels[2] / len(baseline_with_neg) if baseline_with_neg else 0
baseline_pred_contrad_pct = baseline_neg_preds[2] / len(baseline_with_neg) if baseline_with_neg else 0
debiased_pred_contrad_pct = debiased_neg_preds[2] / len(debiased_with_neg) if debiased_without_neg else 0

# Create visualizations for negation analysis
print("\nCreating negation analysis visualizations...")

# Figure: Accuracy comparison and label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Optimized for two-column layout

# Subplot 1: Accuracy comparison
categories = ['With\nNegation', 'Without\nNegation']
baseline_accs = [baseline_neg_acc, baseline_no_neg_acc]
debiased_accs = [debiased_neg_acc, debiased_no_neg_acc]

x = np.arange(len(categories))
width = 0.35

bars1_base = axes[0].bar(x - width/2, baseline_accs, width, label='Baseline', 
                         alpha=0.85, color=colors_palette['baseline'], edgecolor='white', linewidth=1.5)
bars1_deb = axes[0].bar(x + width/2, debiased_accs, width, label='Debiased', 
                        alpha=0.85, color=colors_palette['debiased'], edgecolor='white', linewidth=1.5)

axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
axes[0].set_title('Accuracy: Negation vs Non-Negation', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=10)
axes[0].legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
axes[0].set_ylim([0.8, 0.9])
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add value labels
for bars, accs in [(bars1_base, baseline_accs), (bars1_deb, debiased_accs)]:
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Subplot 2: Label distribution for negation examples
labels = ['Entailment', 'Neutral', 'Contradiction']
true_dist = [neg_true_labels[0]/len(baseline_with_neg),
             neg_true_labels[1]/len(baseline_with_neg),
             neg_true_labels[2]/len(baseline_with_neg)]
baseline_pred_dist = [baseline_neg_preds[0]/len(baseline_with_neg),
                      baseline_neg_preds[1]/len(baseline_with_neg),
                      baseline_neg_preds[2]/len(baseline_with_neg)]
debiased_pred_dist = [debiased_neg_preds[0]/len(debiased_with_neg),
                      debiased_neg_preds[1]/len(debiased_with_neg),
                      debiased_neg_preds[2]/len(debiased_with_neg)]

x2 = np.arange(len(labels))
width2 = 0.25

bars2_true = axes[1].bar(x2 - width2, true_dist, width2, label='True Labels', 
                         alpha=0.85, color=colors_palette['true'], edgecolor='white', linewidth=1.5)
bars2_base = axes[1].bar(x2, baseline_pred_dist, width2, label='Baseline', 
                         alpha=0.85, color=colors_palette['baseline'], edgecolor='white', linewidth=1.5)
bars2_deb = axes[1].bar(x2 + width2, debiased_pred_dist, width2, label='Debiased', 
                        alpha=0.85, color=colors_palette['debiased'], edgecolor='white', linewidth=1.5)

axes[1].set_ylabel('Proportion', fontsize=11, fontweight='bold')
axes[1].set_title('Label Distribution: Hypotheses WITH Negation', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
axes[1].legend(fontsize=9, frameon=True, fancybox=True, shadow=True)
axes[1].set_ylim([0, 0.6])
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add value labels for label distribution
for bars, dists in [(bars2_true, true_dist), (bars2_base, baseline_pred_dist), (bars2_deb, debiased_pred_dist)]:
    for bar, dist in zip(bars, dists):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{dist:.1%}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout(pad=2.0)
output_dir = os.path.join(project_root, 'outputs', 'evaluations')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'negation_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Negation analysis chart saved to: {output_path}")
plt.close()

print("Negation analysis visualizations completed!")

