# Phase 2: 100K Training Results

## 📊 Final Accuracies (100K Training, 9,842 Validation)

| Model | Accuracy | Training Details | Status |
|-------|----------|------------------|--------|
| **Random Baseline** | 33.33% | Random guessing | Theoretical |
| **Hypothesis-Only** | 60.80% | 100K train, hypothesis only | ✅ Complete |
| **Baseline** | 86.54% | 100K train, premise + hypothesis | ✅ Complete |
| **Debiased** | 86.42% | 100K train, artifact reweighting | ✅ Complete |

---

## 🎯 Key Findings

### Artifact Detection
- **Hypothesis-Only Performance:** 60.80%
- **Above Random:** +27.47% (60.80% - 33.33%)
- **Interpretation:** Strong artifacts detected! Model can predict ~61% without seeing the premise.

### Debiasing Effect
- **Baseline Accuracy:** 86.54%
- **Debiased Accuracy:** 86.42%
- **Change:** -0.12%
- **Interpretation:** Minimal overall change (within noise margin). Debiasing may improve robustness on hard examples without sacrificing overall accuracy.

---

## 📝 Training Details

### Model 1: Baseline
```
Command: python run.py --do_train --do_eval --task nli --dataset snli 
         --output_dir ./baseline_100k/ --max_train_samples 100000 
         --num_train_epochs 3 --per_device_train_batch_size 32

Training Time: ~9.5 minutes
Final Loss: 0.3832
Accuracy: 86.54%
File: baseline_100k/eval_predictions.jsonl (9,842 predictions)
```

### Model 2: Hypothesis-Only (Artifact Detector)
```
Command: python train_hypothesis_only.py

Training Time: ~7 minutes
Final Loss: 0.9152
Accuracy: 60.80%
Interpretation: 27.47% above random = STRONG artifacts
File: hypothesis_only_model/eval_predictions.jsonl (9,842 predictions)
```

### Model 3: Debiased
```
Command: python train_debiased.py

Training Time: ~28 minutes (slower due to dual model inference)
Training Loss: 0.2731
Evaluation Loss: 0.2440
Accuracy: 86.42%
Method: Confidence-based reweighting using hypothesis-only model
Formula: weight = 1.0 / (1.0 + bias_confidence)
File: debiased_model/eval_predictions.jsonl (9,842 predictions)
```

---

## 📈 Training Metrics (from Colab Output)

### Baseline Model
```
Epoch 1: loss ~0.53
Epoch 2: loss ~0.40
Epoch 3: loss ~0.34
Final: eval_loss=0.3832, eval_accuracy=0.8654
```

### Hypothesis-Only Model
```
Epoch 1: loss ~1.03, eval_accuracy=0.528
Epoch 2: loss ~0.91, eval_accuracy=0.600
Epoch 3: loss ~0.84, eval_accuracy=0.608
Final: eval_loss=0.9152, eval_accuracy=0.6080
```

### Debiased Model
```
Training: 18,750 steps total
Train loss: 0.2731 (lower than baseline!)
Eval loss: 0.2440 (lower than baseline!)
Accuracy: 0.8642 (86.42%)
```

---

## 🎯 For Your Paper

### Table 1: Main Results
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
Model & Accuracy & $\Delta$ vs Baseline \\
\hline
Random & 33.33\% & -53.21\% \\
Hypothesis-Only & 60.80\% & -25.74\% \\
Baseline & 86.54\% & - \\
Debiased & 86.42\% & -0.12\% \\
\hline
\end{tabular}
\caption{Model performance on SNLI validation set (9,842 examples)}
\end{table}
```

### Key Points for Discussion

1. **Strong Artifacts Exist:**
   - Hypothesis-only model achieves 60.80% (vs 33.33% random)
   - 27.47 percentage points above chance
   - Confirms significant dataset artifacts in SNLI

2. **Debiasing Trade-off:**
   - Minimal accuracy change (-0.12%)
   - Suggests artifacts not heavily relied upon by baseline
   - OR: Debiasing maintains accuracy while reducing artifact dependence

3. **Next Steps:**
   - Error analysis: Which examples did debiasing fix/break?
   - Per-class analysis: Did neutral class improve?
   - Negation examples: Did debiasing help with negation bias?

---

## 📁 File Locations

```
Final/fp-dataset-artifacts/
├── baseline_100k/
│   └── eval_predictions.jsonl (86.54% accuracy)
├── hypothesis_only_model/
│   └── eval_predictions.jsonl (60.80% accuracy)
└── debiased_model/
    └── eval_predictions.jsonl (86.42% accuracy)
```

All files: 9,842 predictions each (full SNLI validation set)

---

## ⏱️ Total Time Invested

- Setup and debugging: ~1 hour
- Training time: ~45 minutes (9.5 + 7 + 28)
- **Total Phase 2A:** ~1.75 hours

---

## 🚀 Next Phase: Analysis

### TODO:
1. Run `error_analysis.py` on baseline
2. Run `compare_models.py` to compare baseline vs debiased
3. Identify specific improvements and failures
4. Create visualizations for paper

---

*Results recorded: November 8, 2024*
*Phase 2A: Training Complete ✅*
*Next: Phase 2B - Analysis*

