import json
from collections import defaultdict, Counter
import re

# Load predictions
print("Loading predictions...")
predictions = []
# Update this path to wherever your eval_predictions.jsonl file is
predictions_path = r'c:\Users\dinah\Downloads\eval_predictions.jsonl'
with open(predictions_path, 'r', encoding='utf-8') as f:
    for line in f:
        predictions.append(json.loads(line))

print(f"Total examples: {len(predictions)}\n")

# Calculate overall accuracy
correct = sum(1 for p in predictions if p['label'] == p['predicted_label'])
accuracy = correct / len(predictions)
print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
print("=" * 80)

# Separate correct and incorrect predictions
errors = [p for p in predictions if p['label'] != p['predicted_label']]
correct_preds = [p for p in predictions if p['label'] == p['predicted_label']]

print(f"\nCorrect predictions: {len(correct_preds)}")
print(f"Incorrect predictions: {len(errors)} ({len(errors)/len(predictions):.1%})")
print("=" * 80)

# Label distribution
label_names = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
print("\n=== LABEL DISTRIBUTION ===")
true_labels = Counter(p['label'] for p in predictions)
for label, count in sorted(true_labels.items()):
    print(f"{label_names[label]}: {count} ({count/len(predictions):.1%})")

# Confusion matrix
print("\n=== CONFUSION MATRIX ===")
print("Rows = True Label, Columns = Predicted Label")
print(f"{'':20} {'Entail':>10} {'Neutral':>10} {'Contrad':>10}")
confusion = defaultdict(lambda: defaultdict(int))
for p in predictions:
    confusion[p['label']][p['predicted_label']] += 1

for true_label in [0, 1, 2]:
    row = f"{label_names[true_label]:20}"
    for pred_label in [0, 1, 2]:
        count = confusion[true_label][pred_label]
        row += f"{count:>10}"
    print(row)

# Per-class accuracy
print("\n=== PER-CLASS ACCURACY ===")
for label in [0, 1, 2]:
    total = true_labels[label]
    correct_for_label = confusion[label][label]
    acc = correct_for_label / total if total > 0 else 0
    print(f"{label_names[label]:15}: {acc:.2%} ({correct_for_label}/{total})")

print("\n" + "=" * 80)
print("=== HYPOTHESIS-ONLY ARTIFACT ANALYSIS ===")
print("Testing if model learns patterns from hypothesis words alone...")

# Check for negation words in hypothesis
negation_words = ['no', 'not', 'never', 'nobody', 'nothing', 'nowhere', 'neither', 'none', "n't"]

def has_negation(text):
    text_lower = text.lower()
    return any(neg in text_lower for neg in negation_words)

# Analyze negation correlation with contradiction
hyp_with_negation = [p for p in predictions if has_negation(p['hypothesis'])]
hyp_without_negation = [p for p in predictions if not has_negation(p['hypothesis'])]

print(f"\nHypotheses with negation words: {len(hyp_with_negation)}")
print(f"Hypotheses without negation: {len(hyp_without_negation)}")

if hyp_with_negation:
    neg_labels = Counter(p['label'] for p in hyp_with_negation)
    print(f"\nTrue label distribution for hypotheses WITH negation:")
    for label, count in sorted(neg_labels.items()):
        print(f"  {label_names[label]}: {count} ({count/len(hyp_with_negation):.1%})")
    
    neg_preds = Counter(p['predicted_label'] for p in hyp_with_negation)
    print(f"\nPredicted label distribution for hypotheses WITH negation:")
    for label, count in sorted(neg_preds.items()):
        print(f"  {label_names[label]}: {count} ({count/len(hyp_with_negation):.1%})")

print("\n" + "=" * 80)
print("=== EXAMPLE ERRORS ===\n")

# Show some interesting errors
print("1. Examples where TRUE=Neutral but PREDICTED=Contradiction:")
neutral_to_contra = [p for p in errors if p['label'] == 1 and p['predicted_label'] == 2]
for i, ex in enumerate(neutral_to_contra[:3], 1):
    print(f"Error {i}:")
    print(f"  Premise: {ex['premise']}")
    print(f"  Hypothesis: {ex['hypothesis']}")
    print(f"  True: Neutral, Predicted: Contradiction")
    print()

print("2. Examples where TRUE=Contradiction but PREDICTED=Neutral:")
contra_to_neutral = [p for p in errors if p['label'] == 2 and p['predicted_label'] == 1]
for i, ex in enumerate(contra_to_neutral[:3], 1):
    print(f"Error {i}:")
    print(f"  Premise: {ex['premise']}")
    print(f"  Hypothesis: {ex['hypothesis']}")
    print(f"  True: Contradiction, Predicted: Neutral")
    print()

print("3. Examples where TRUE=Entailment but PREDICTED=Neutral:")
entail_to_neutral = [p for p in errors if p['label'] == 0 and p['predicted_label'] == 1]
for i, ex in enumerate(entail_to_neutral[:3], 1):
    print(f"Error {i}:")
    print(f"  Premise: {ex['premise']}")
    print(f"  Hypothesis: {ex['hypothesis']}")
    print(f"  True: Entailment, Predicted: Neutral")
    print()

print("=" * 80)
print("\n=== SUMMARY ===")
print(f"Total errors: {len(errors)}")
print(f"Most common error types:")
error_types = Counter((p['label'], p['predicted_label']) for p in errors)
for (true_l, pred_l), count in error_types.most_common(5):
    print(f"  True={label_names[true_l]:13} -> Predicted={label_names[pred_l]:13}: {count:4} ({count/len(errors):.1%})")

print("\n" + "=" * 80)
print("Analysis complete! Now let's discuss what we found...")

