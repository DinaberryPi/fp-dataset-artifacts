# Download Checklist from Colab

## 📥 Files to Download (Small, Git-Friendly)

### ✅ Essential Files (Already Downloaded)
- [x] `baseline_100k/eval_predictions.jsonl` (2.5 MB)
- [x] `hypothesis_only_model/eval_predictions.jsonl` (2.5 MB)
- [x] `debiased_model/eval_predictions.jsonl` (2.4 MB)

### 📊 Metrics Files (SHOULD DOWNLOAD - Small!)

Run in Colab:
```python
from google.colab import files

# Download metrics files (contains accuracy, loss, etc.)
files.download('baseline_100k/eval_metrics.json')
files.download('hypothesis_only_model/eval_metrics.json')
files.download('debiased_model/eval_metrics.json')
```

**What's in these files:**
- Exact accuracy numbers
- Evaluation loss
- Runtime statistics
- Samples per second
- Official record for your paper!

**Size:** ~1-2 KB each (tiny!)

---

### 🗂️ Optional: Training Logs (if you want complete records)

Run in Colab:
```python
# Training state and logs
files.download('baseline_100k/trainer_state.json')
files.download('hypothesis_only_model/trainer_state.json')
files.download('debiased_model/trainer_state.json')
```

**What's in these files:**
- Training history (loss per step)
- Learning rate schedule
- Best checkpoint info
- Complete training timeline

**Size:** ~50-100 KB each

---

### 📈 WandB Logs (Optional - for graphs)

You have wandb runs saved:
```
wandb/offline-run-20251108_025835-fs4yh5gl  (baseline)
wandb/offline-run-20251108_041435-1gekld2x  (hypothesis-only)
wandb/offline-run-20251108_053335-93myf7es  (debiased)
```

To download:
```python
# Zip the wandb folder
!zip -r wandb_logs.zip wandb/

# Download
files.download('wandb_logs.zip')
```

**What's in these:**
- Training curves (loss over time)
- System metrics (GPU usage, etc.)
- Can visualize later with `wandb sync`

**Size:** ~5-10 MB

---

## 🎯 Recommended: Download Now

**Minimum (for your paper):**
```python
from google.colab import files

# Quick download - just metrics
files.download('baseline_100k/eval_metrics.json')
files.download('hypothesis_only_model/eval_metrics.json')
files.download('debiased_model/eval_metrics.json')
```

**These 3 tiny files contain all the accuracy/loss numbers you need!**

---

## 📁 Where to Save Locally

Save to your project folder:
```
C:\Users\dinah\Desktop\Code\NLP\Final\fp-dataset-artifacts\
├── baseline_100k\
│   ├── eval_metrics.json          ← Download this!
│   └── eval_predictions.jsonl     ← Already have
├── hypothesis_only_model\
│   ├── eval_metrics.json          ← Download this!
│   └── eval_predictions.jsonl     ← Already have
└── debiased_model\
    ├── eval_metrics.json          ← Download this!
    └── eval_predictions.jsonl     ← Already have
```

---

## 🔍 After Downloading

Run this to verify and display your metrics:
```bash
python read_metrics.py
```

This will show:
- All three model accuracies
- Artifact strength calculation
- Debiasing effect
- Complete metrics breakdown

---

## 💾 Already Saved (Manual)

You already have these manual records:
- ✅ `TRAINING_RESULTS.md` - Clean summary
- ✅ `colab_output.txt` - Raw Colab output

But the `eval_metrics.json` files are the **official source of truth**!

---

## Summary

**Must Download (3 files, <5 KB total):**
- [ ] `baseline_100k/eval_metrics.json`
- [ ] `hypothesis_only_model/eval_metrics.json`
- [ ] `debiased_model/eval_metrics.json`

**Already Have:**
- [x] All prediction files
- [x] Manual result records

**Optional:**
- [ ] `trainer_state.json` files (training history)
- [ ] `wandb` logs (training curves)

