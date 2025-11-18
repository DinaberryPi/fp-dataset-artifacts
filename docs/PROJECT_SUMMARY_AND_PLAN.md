# Dataset Artifacts Project - Summary & Next Steps

## 📊 Project Overview

**Goal**: Analyze and mitigate dataset artifacts in NLI (Natural Language Inference) using SNLI dataset

**Method**: Ensemble debiasing with hypothesis-only artifact model

---

## ✅ What We've Accomplished (Phase 1: 5K Training)

### 1. Baseline Model
- **Training**: 5,000 examples from SNLI
- **Evaluation**: 9,842 validation examples
- **Accuracy**: 76.09%
- **Status**: ✅ Complete
- **Files**: 
  - Predictions: `c:\Users\dinah\Downloads\eval_predictions.jsonl` (9,842 lines)
  - Model: In Colab at `./trained_model/`

### 2. Error Analysis
- **Found**: Neutral class weakest (68.04% accuracy vs 81% entailment, 79% contradiction)
- **Found**: Over-predicts contradiction (771 errors where True=Neutral → Predicted=Contradiction)
- **Found**: Negation bias detected (hypothesis with negation words)
- **Status**: ✅ Complete
- **Script**: `error_analysis.py`

### 3. Hypothesis-Only Model (Artifact Detector)
- **Training**: 5,000 examples, sees ONLY hypothesis (not premise)
- **Accuracy**: 59.40% (vs 33.3% random guessing!)
- **Key Finding**: 26 percentage points above random = STRONG artifacts!
- **Status**: ✅ Complete
- **Files**: Model saved to `./hypothesis_only_model/`

### 4. Debiased Model
- **Method**: Confidence-based reweighting
  - Uses hypothesis-only model to identify artifact examples
  - Downweights examples where bias model is confident
  - Formula: `weight = 1.0 / (1.0 + bias_confidence)`
- **Training**: 5,000 examples
- **Evaluation**: 500 examples (⚠️ INCOMPLETE - need full 9,842!)
- **Accuracy**: 79.80% on 500 examples
- **Status**: ⚠️ Partial - needs full evaluation
- **Files**: `c:\Users\dinah\Desktop\Code\NLP\Final\fp-dataset-artifacts\debiase\eval_predictions_debiase.jsonl` (500 lines)

### 5. Comparison Results (on 500 examples)
- **Baseline**: 80.40%
- **Debiased**: 79.80%
- **Change**: -0.60% (essentially tied)
- **Per-class improvements**:
  - Entailment: +1.78% ✅
  - Neutral: -1.26%
  - Contradiction: -2.33%
- **Qualitative improvements**: Fixed 24 examples, broke 27 (net -3)
  - But fixes show more careful reasoning (better neutral recognition)

---

## 🚨 Critical Issue Discovered

**Problem**: Unfair comparison!
- Baseline evaluated on: 9,842 examples (76.09%)
- Debiased evaluated on: 500 examples (79.80%)
- Cannot compare! Different datasets!

**The first 500 examples are EASIER**:
- Baseline on 500: 80.40%
- Baseline on full 9,842: 76.09%
- Difference: +4.31% (500 is easier subset!)

**Solution**: Need to evaluate debiased model on full 9,842 examples
- Script created: `evaluate_full.py`
- Status: Not yet run

---

## 🎯 Next Steps: Scale to 100K Training

### Why 100K instead of 5K?

1. **More robust baseline**: 5K → 76%, but 100K should → ~87% (per README)
2. **More convincing**: Matches standard SNLI baselines
3. **Better for paper**: Example papers use thorough experiments
4. **More reliable artifacts**: Larger data = clearer patterns
5. **Still manageable**: ~30 mins per model (90 mins total)

---

## 📋 TODO List for 100K Training

### Phase 2A: Train New Models (in Colab)

**Files to upload to Colab first:**
- `helpers.py` (updated with hypothesis-only function)
- `run.py` (original training script)
- `train_hypothesis_only.py`
- `train_debiased.py`

#### TODO 1: Train Baseline on 100K ⏱️ 30 minutes

```bash
cd fp-dataset-artifacts

python run.py --do_train --do_eval --task nli --dataset snli \
  --output_dir ./baseline_100k/ \
  --max_train_samples 100000 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32
```

**Expected result**: ~87% accuracy on full validation set

**Download**: 
- `baseline_100k/eval_predictions.jsonl` (should be 9,842 lines)
- Save to local machine

---

#### TODO 2: Train Hypothesis-Only on 100K ⏱️ 30 minutes

**First, update `train_hypothesis_only.py`:**
Change line 76:
```python
max_train_samples = 100000  # Was 5000
```

```bash
python train_hypothesis_only.py
```

**Expected result**: ~60-65% accuracy (still above random!)

**Note**: Model auto-saves to `./hypothesis_only_model/` (will overwrite 5K version)

---

#### TODO 3: Train Debiased on 100K ⏱️ 30 minutes

**First, update `train_debiased.py`:**
Change lines 88-89:
```python
max_train_samples = 100000  # Was 5000
max_eval_samples = None      # Was 500 - now evaluate on FULL set!
```

```bash
python train_debiased.py
```

**Expected result**: ~87-89% accuracy on full validation set

**Download**:
- `debiased_model/eval_predictions.jsonl` (should be 9,842 lines)
- Save to local machine

---

### Phase 2B: Analysis & Comparison

#### TODO 4: Re-run Error Analysis

Update `error_analysis.py` to point to new baseline predictions:
```python
predictions_path = r'c:\Users\dinah\Downloads\baseline_100k_predictions.jsonl'
```

```bash
python error_analysis.py
```

**Look for**:
- Is neutral still weakest class?
- What's the negation bias percentage now?
- Are error patterns the same?

---

#### TODO 5: Run Comparison

Update `compare_models.py` paths:
```python
baseline_path = r'c:\Users\dinah\Downloads\baseline_100k_predictions.jsonl'
debiased_path = r'c:\Users\dinah\Downloads\debiased_100k_predictions.jsonl'
```

```bash
python compare_models.py
```

**Key metrics to report**:
- Overall accuracy change
- Per-class improvements
- Negation example improvements
- Number of fixes vs breaks
- Example fixes to showcase

---

### Phase 2C: Write Paper

#### TODO 6: Structure Paper (ACL Format)

**Sections to write**:

1. **Abstract** (150-200 words)
   - Problem: Dataset artifacts in NLI
   - Method: Hypothesis-only debiasing
   - Results: X% baseline → Y% debiased
   - Key finding: Artifacts comprise Z% of performance

2. **Introduction** (1 page)
   - NLI task importance
   - Dataset artifacts problem
   - Your approach overview
   - Contributions

3. **Related Work** (0.5 pages)
   - Cite papers from spec: Poliak et al. 2018, He et al. 2019, Clark et al. 2019
   - Dataset cartography, ensemble debiasing methods

4. **Approach** (1-1.5 pages)
   - **4.1 Hypothesis-Only Model**: Explain artifact detection
   - **4.2 Debiasing Method**: Confidence-based reweighting
   - **4.3 Training Details**: ELECTRA-small, 100K samples, 3 epochs

5. **Experiments** (0.5 pages)
   - Dataset: SNLI (100K train, 9,842 validation)
   - Models: Baseline, hypothesis-only, debiased
   - Evaluation: Accuracy, per-class, artifact examples

6. **Results** (1.5-2 pages)
   - **Table 1**: Overall accuracies
   - **Table 2**: Per-class accuracies
   - **Table 3**: Negation example accuracies
   - **Table 4**: Confusion matrices
   - **Examples**: Show 3-5 fixes from debiased model
   - **Analysis**: Why improvements occurred

7. **Discussion** (0.5-1 page)
   - What artifacts were found
   - How debiasing helped
   - Where it didn't help (if any)
   - Limitations

8. **Conclusion** (0.5 pages)
   - Summary of findings
   - Future work: Scale to other datasets, try other debiasing methods

---

## 📊 Expected Results to Report

### Table 1: Main Results
| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Random Baseline | 33.3% | - |
| Hypothesis-Only | ~62% | +29% over random |
| Full Baseline | ~87% | - |
| Debiased | ~88-89% | +1-2% |

### Table 2: Per-Class Accuracy
| Class | Baseline | Debiased | Change |
|-------|----------|----------|--------|
| Entailment | ~87% | ~88% | +1% |
| Neutral | ~84% | ~86% | +2% |
| Contradiction | ~89% | ~90% | +1% |

*(These are estimates - actual numbers from your training)*

---

## 📁 File Locations Reference

### Local Machine
```
c:\Users\dinah\Desktop\Code\NLP\Final\fp-dataset-artifacts\
├── helpers.py (updated with hypothesis-only function)
├── run.py (baseline training)
├── train_hypothesis_only.py
├── train_debiased.py
├── error_analysis.py
├── compare_models.py
├── evaluate_full.py
└── debiase\
    └── eval_predictions_debiase.jsonl (5K results - 500 lines)

c:\Users\dinah\Downloads\
└── eval_predictions.jsonl (5K baseline - 9,842 lines)
```

### Colab
```
/content/fp-dataset-artifacts/
├── baseline_100k/ (after TODO 1)
├── hypothesis_only_model/ (after TODO 2)
└── debiased_model/ (after TODO 3)
```

---

## 🎓 Key Concepts for Paper

### What are Dataset Artifacts?
Spurious correlations that allow models to achieve high accuracy without true understanding.

**Example in SNLI**: Hypotheses with "nobody" are often contradictions → model learns "nobody" → contradiction shortcut.

### Why Hypothesis-Only Model?
- If it gets >33% (random), artifacts exist!
- Yours got 59.4% on 5K → strong artifacts
- Quantifies how much performance comes from shortcuts

### How Debiasing Works?
1. Hypothesis-only model identifies artifact examples (high confidence = artifact)
2. Downweight these examples: `weight = 1 / (1 + confidence)`
3. Force main model to learn from hard examples requiring premise

**Example**:
```
Hypothesis: "Nobody is happy"
Hypothesis-only: 85% confident → Contradiction (artifact!)
Weight: 1/(1+0.85) = 0.54 (reduced from 1.0)
Effect: Model learns less from this, focuses on premise instead
```

---

## 💡 Tips for New Chat

**When you start a new chat, share:**
1. This summary file
2. Your current results (accuracies, tables)
3. What phase you're on (training? writing?)

**Quick context for AI:**
> "I'm working on an NLP project analyzing dataset artifacts in SNLI. I've trained baseline (76%), hypothesis-only (59%), and debiased (79.8%) models on 5K examples. Now scaling to 100K for stronger results. See PROJECT_SUMMARY_AND_PLAN.md for details."

---

## 📚 References to Include in Paper

- Bowman et al., 2015 - SNLI dataset
- Poliak et al., 2018 - Hypothesis-only baselines
- Gardner et al., 2020 - Contrast sets
- He et al., 2019 - Ensemble debiasing
- Clark et al., 2019 - Product of Experts
- Clark et al., 2020 - ELECTRA model

---

## ⚡ Quick Commands Cheat Sheet

**In Colab:**
```bash
# Setup
%cd fp-dataset-artifacts

# Train baseline 100K
!python run.py --do_train --do_eval --task nli --dataset snli --output_dir ./baseline_100k/ --max_train_samples 100000 --num_train_epochs 3 --per_device_train_batch_size 32

# Train hypothesis-only 100K (after updating script)
!python train_hypothesis_only.py

# Train debiased 100K (after updating script)
!python train_debiased.py

# Download results
from google.colab import files
files.download('baseline_100k/eval_predictions.jsonl')
files.download('debiased_model/eval_predictions.jsonl')
```

**On Local Machine:**
```bash
cd "c:\Users\dinah\Desktop\Code\NLP\Final\fp-dataset-artifacts"

# Analysis
python error_analysis.py

# Comparison
python compare_models.py
```

---

## 🎯 Success Criteria

**Minimum for good grade:**
- ✅ Clear artifact identification (hypothesis-only > random)
- ✅ Implemented debiasing method
- ✅ Showed some improvement (even if small)
- ✅ Good analysis and discussion
- ✅ Well-written paper (3-8 pages)

**Your project status**: On track! Just need to scale up and write.

---

## 📞 Questions for Next Session?

- Results from 100K training?
- Help analyzing new results?
- Paper writing assistance?
- Creating tables/figures?
- Understanding specific results?

---

**Estimated Time Remaining:**
- Training (100K × 3 models): 1.5-2 hours
- Analysis: 30 minutes  
- Paper writing: 4-6 hours
- **Total**: 6-9 hours of work

**You've got this!** 🚀

---

*Generated: November 7, 2024*
*Progress: Phase 1 Complete (5K training) → Next: Phase 2 (100K training)*

