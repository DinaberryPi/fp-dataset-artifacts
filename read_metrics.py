"""
Read and display all eval_metrics.json files from training.
"""
import json
import os

def read_metrics(model_dir, model_name):
    """Read and display metrics from a model directory"""
    metrics_path = os.path.join(model_dir, 'eval_metrics.json')
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        
        # Key metrics
        if 'eval_accuracy' in metrics:
            print(f"Accuracy:          {metrics['eval_accuracy']*100:.2f}%")
        if 'eval_loss' in metrics:
            print(f"Loss:              {metrics['eval_loss']:.4f}")
        if 'eval_runtime' in metrics:
            print(f"Eval Runtime:      {metrics['eval_runtime']:.1f} seconds")
        if 'eval_samples_per_second' in metrics:
            print(f"Samples/Second:    {metrics['eval_samples_per_second']:.1f}")
        if 'epoch' in metrics:
            print(f"Epochs:            {metrics['epoch']:.0f}")
        
        # Full metrics
        print(f"\nFull metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:30s}: {value:.6f}")
            else:
                print(f"  {key:30s}: {value}")
        
        return metrics
    else:
        print(f"\n{'='*70}")
        print(f"{model_name}")
        print('='*70)
        print(f"[WARNING] File not found: {metrics_path}")
        return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING METRICS - PHASE 2 (100K)")
    print("="*70)
    
    models = [
        ('baseline_100k', 'Baseline Model (100K)'),
        ('hypothesis_only_model', 'Hypothesis-Only Model (Artifact Detector)'),
        ('debiased_model', 'Debiased Model (100K)')
    ]
    
    all_metrics = {}
    
    for model_dir, model_name in models:
        metrics = read_metrics(model_dir, model_name)
        if metrics:
            all_metrics[model_name] = metrics
    
    # Summary comparison
    if len(all_metrics) >= 2:
        print(f"\n{'='*70}")
        print("SUMMARY COMPARISON")
        print('='*70)
        
        for model_name in all_metrics:
            acc = all_metrics[model_name].get('eval_accuracy', 0) * 100
            loss = all_metrics[model_name].get('eval_loss', 0)
            print(f"{model_name:45s}: {acc:6.2f}% (loss: {loss:.4f})")
        
        # Calculate artifact strength and debiasing effect
        if 'Hypothesis-Only Model (Artifact Detector)' in all_metrics:
            hyp_acc = all_metrics['Hypothesis-Only Model (Artifact Detector)']['eval_accuracy'] * 100
            artifact_strength = hyp_acc - 33.33
            print(f"\nArtifact Strength: +{artifact_strength:.2f}% above random guessing")
        
        if 'Baseline Model (100K)' in all_metrics and 'Debiased Model (100K)' in all_metrics:
            baseline_acc = all_metrics['Baseline Model (100K)']['eval_accuracy'] * 100
            debiased_acc = all_metrics['Debiased Model (100K)']['eval_accuracy'] * 100
            improvement = debiased_acc - baseline_acc
            print(f"Debiasing Effect:  {improvement:+.2f}% change")
            
            if abs(improvement) < 0.5:
                print("                   (Minimal change - maintains performance)")
            elif improvement > 0:
                print("                   (Improvement!)")
            else:
                print("                   (May improve robustness despite lower accuracy)")
        
        print("\n" + "="*70)

