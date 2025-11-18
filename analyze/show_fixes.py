"""
Show examples where debiasing fixed baseline errors.
"""
import json
import os

# Get the project root directory (parent of analyze/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

# Load fixes
fixes_file = os.path.join(project_root, 'outputs', 'evaluations', 'fixes_examples.json')
if not os.path.exists(fixes_file):
    print("Error: fixes_examples.json not found. Please run compare_results.py first.")
    exit(1)

with open(fixes_file, 'r', encoding='utf-8') as f:
    fixes = json.load(f)

print("=" * 80)
print("Examples Where Debiasing Fixed Baseline Errors")
print("=" * 80)

for i, fix in enumerate(fixes[:5], 1):
    print(f"\nFix Example {i}:")
    print(f"  Premise: {fix['premise']}")
    print(f"  Hypothesis: {fix['hypothesis']}")
    print(f"  True Label: {label_names[fix['true_label']]}")
    print(f"  Baseline Predicted: {label_names[fix['baseline_pred']]} [WRONG]")
    print(f"  Debiased Predicted: {label_names[fix['debiased_pred']]} [CORRECT]")
    print("-" * 80)

