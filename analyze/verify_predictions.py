"""
Quick script to verify your predictions file is correct
"""
import json

def verify_predictions(filepath):
    """Check if predictions file is valid"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        num_lines = len(lines)
        print(f"✅ File found: {filepath}")
        print(f"✅ Number of examples: {num_lines}")
        
        # Expected: 9842 for full validation set
        if num_lines == 9842:
            print("✅ PERFECT! Full validation set (9,842 examples)")
        else:
            print(f"⚠️  Expected 9,842 examples, got {num_lines}")
        
        # Check first example format
        first = json.loads(lines[0])
        print(f"\n📋 First example keys: {list(first.keys())}")
        
        required_keys = ['premise', 'hypothesis', 'label', 'predicted_label', 'predicted_scores']
        missing = [k for k in required_keys if k not in first]
        
        if not missing:
            print("✅ All required fields present")
            print(f"\nExample:")
            print(f"  Premise: {first['premise'][:60]}...")
            print(f"  Hypothesis: {first['hypothesis'][:60]}...")
            print(f"  True label: {first['label']}")
            print(f"  Predicted label: {first['predicted_label']}")
        else:
            print(f"❌ Missing fields: {missing}")
            
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Test with your file
if __name__ == "__main__":
    # Update this path after you download from Colab
    predictions_path = r'C:\Users\dinah\Downloads\baseline_100k_predictions.jsonl'
    verify_predictions(predictions_path)

