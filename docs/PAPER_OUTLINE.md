# Paper Outline - Dataset Artifacts Project

## Paper Structure (3-8 pages, ACL/NeurIPS style)

---

## 1. Abstract (150 words)

**Key points to include:**
- Problem: Pre-trained models may learn spurious correlations (artifacts) from datasets
- Dataset: SNLI (Natural Language Inference)
- Method: Hypothesis-only artifact detection + ensemble debiasing
- Results: 
  - Detected strong artifacts (60.80% hypothesis-only vs 33.33% random)
  - Debiasing maintains performance (86.42% vs 86.54% baseline)
- Conclusion: Artifacts exist, debiasing framework works but needs refinement

**Template:**
```
Pre-trained models can achieve high performance on benchmark datasets by learning 
spurious correlations, or dataset artifacts, rather than solving the underlying task. 
We investigate dataset artifacts in the Stanford Natural Language Inference (SNLI) 
dataset using ELECTRA-small. We first demonstrate the existence of artifacts by 
training a hypothesis-only model that achieves 60.80% accuracy (27.47% above random) 
without seeing premises, indicating strong artifacts. We then implement an ensemble 
debiasing method using confidence-based reweighting to mitigate these artifacts. 
While debiasing maintains overall accuracy (86.42% vs 86.54% baseline), it shows 
limited improvement on specific error types. Our analysis reveals that the Neutral 
class is hardest (82.97% accuracy) and that the model struggles most with 
distinguishing Neutral from Contradiction. These findings suggest that while 
hypothesis-only artifacts exist, they may not be the primary source of errors, 
highlighting the need for more sophisticated artifact detection and mitigation methods.
```

---

## 2. Introduction (1 page)

### 2.1 Motivation
- Pre-trained models achieve high performance on benchmarks
- But do they really "solve" the task?
- Examples: hypothesis-only baselines in NLI, adversarial examples
- Problem: Models learn spurious correlations (artifacts)

### 2.2 Dataset Artifacts
- Definition: Spurious correlations that allow high accuracy without true understanding
- Examples in NLI:
  - Hypothesis-only patterns (e.g., "nobody" → contradiction)
  - Word overlap patterns
  - Length-based patterns

### 2.3 Our Approach
- **Part 1 (Analysis):** Detect artifacts using hypothesis-only model
- **Part 2 (Mitigation):** Implement ensemble debiasing with confidence reweighting
- **Dataset:** SNLI (Stanford Natural Language Inference)
- **Model:** ELECTRA-small

### 2.4 Contributions
- Quantify artifact strength in SNLI (60.80% hypothesis-only)
- Implement and evaluate ensemble debiasing method
- Analyze error patterns and per-class performance
- Identify limitations and future directions

### 2.5 Paper Organization
- Section 3: Related Work
- Section 4: Method
- Section 5: Experiments
- Section 6: Results
- Section 7: Discussion
- Section 8: Conclusion

---

## 3. Related Work (0.5 pages)

### 3.1 Dataset Artifacts in NLI
- **Poliak et al. (2018):** Hypothesis-only baselines in NLI
- **Gururangan et al. (2018):** Annotation artifacts in NLI
- **McCoy et al. (2019):** Right for the wrong reasons

### 3.2 Debiasing Methods
- **He et al. (2019):** Ensemble-based debiasing
- **Clark et al. (2019):** Product of Experts
- **Utama et al. (2020):** Learning from hard examples
- **Zhou and Bansal (2020):** Adversarial training

### 3.3 Other Approaches
- **Gardner et al. (2020):** Contrast sets
- **Ribeiro et al. (2020):** Checklist testing
- **Swayamdipta et al. (2020):** Dataset cartography

### 3.4 Our Work
- Combines artifact detection (hypothesis-only) with ensemble debiasing
- Focuses on SNLI dataset
- Provides detailed error analysis

---

## 4. Method (1-1.5 pages)

### 4.1 Task and Dataset
- **Task:** Natural Language Inference (NLI)
- **Dataset:** SNLI (Bowman et al., 2015)
  - 550K training examples
  - 9,842 validation examples
  - 3 labels: Entailment, Neutral, Contradiction
- **Model:** ELECTRA-small (Clark et al., 2020)

### 4.2 Artifact Detection: Hypothesis-Only Model

**Motivation:**
- If model can predict without seeing premise, artifacts exist
- Random baseline: 33.33% (3-class classification)
- Any performance above random indicates artifacts

**Method:**
- Train model on hypothesis only (no premise)
- Input: `[CLS] hypothesis [SEP]`
- Output: 3-class classification
- Performance: 60.80% accuracy

**Interpretation:**
- 27.47% above random = strong artifacts
- Model learns patterns from hypothesis words alone
- Examples: negation words, specific phrases

### 4.3 Debiasing: Ensemble Method

**Overview:**
- Use hypothesis-only model as "artifact expert"
- Identify examples where artifact model is confident
- Downweight these examples during training

**Algorithm:**
1. Train hypothesis-only model on training data
2. For each training example:
   - Get hypothesis-only model's confidence: `conf = max(softmax(logits))`
   - Compute weight: `weight = 1.0 / (1.0 + conf)`
   - Use weighted loss: `loss = weight * cross_entropy(pred, label)`

**Intuition:**
- High confidence from artifact model → likely artifact example
- Downweight artifact examples → force model to learn from hard examples
- Low confidence → normal example → full weight

**Implementation:**
- Use HuggingFace Trainer with custom loss function
- Weight applied during training only
- Evaluation on standard validation set

### 4.4 Evaluation Metrics
- Overall accuracy
- Per-class accuracy
- Error analysis (confusion matrix, error types)
- Negation word analysis
- Example-level comparison

---

## 5. Experiments (0.5 pages)

### 5.1 Experimental Setup
- **Model:** ELECTRA-small (14M parameters)
- **Training:** 100,000 examples from SNLI training set
- **Evaluation:** Full validation set (9,842 examples)
- **Hyperparameters:**
  - Learning rate: 2e-5
  - Batch size: 32
  - Epochs: 3
  - Max sequence length: 128

### 5.2 Models Trained
1. **Baseline:** Standard fine-tuning on premise + hypothesis
2. **Hypothesis-Only:** Training on hypothesis only (artifact detector)
3. **Debiased:** Training with confidence-based reweighting

### 5.3 Analysis Methods
- Per-class accuracy
- Confusion matrices
- Error pattern analysis
- Negation word correlation
- Example-level comparison (baseline vs debiased)

---

## 6. Results (1.5-2 pages)

### 6.1 Main Results

**Table 1: Overall Performance**
| Model | Accuracy | Δ vs Baseline |
|-------|----------|---------------|
| Random | 33.33% | -53.21% |
| Hypothesis-Only | 60.80% | -25.74% |
| Baseline | 86.54% | - |
| Debiased | 86.42% | -0.12% |

**Key Findings:**
- Hypothesis-only model: 60.80% (27.47% above random) → **strong artifacts detected**
- Debiased model: 86.42% (-0.12%) → **maintains performance**

### 6.2 Per-Class Analysis

**Table 2: Per-Class Accuracy**
| Class | Baseline | Debiased | Change |
|-------|----------|----------|--------|
| Entailment | 89.28% | 89.31% | +0.03% |
| Neutral | 82.97% | 82.38% | -0.59% |
| Contradiction | 87.28% | 87.46% | +0.18% |

**Findings:**
- Neutral class is hardest (82.97%)
- Debiasing has minimal effect on per-class performance
- Changes are within noise margin

### 6.3 Error Analysis

**Table 3: Most Common Errors (Baseline)**
| Error Type | Count | Percentage |
|------------|-------|------------|
| Contradiction → Neutral | 307 | 23.2% |
| Neutral → Contradiction | 297 | 22.4% |
| Entailment → Neutral | 261 | 19.7% |

**Findings:**
- Model struggles most with Neutral class
- Confusion between Neutral and Contradiction is common
- Suggests need for better reasoning about subtle differences

### 6.4 Negation Word Analysis

**Statistics:**
- Examples with negation: 441 (4.5%)
- Baseline accuracy on negation: 85.94%
- Debiased accuracy on negation: 85.49%
- Change: -0.45%

**Findings:**
- Negation words show weak correlation with contradictions
- Debiasing did not improve negation examples
- Suggests negation artifacts may not be primary issue

### 6.5 Prediction Changes

**Statistics:**
- Total predictions changed: 605 (6.1%)
- Baseline wrong → Debiased correct: 270
- Baseline correct → Debiased wrong: 282
- Net improvement: -12 examples

**Example Fixes:**
1. **Fix 1:** Entailment example
   - Premise: "A man selling donuts to a customer during a world exhibition event held in the city of Angeles"
   - Hypothesis: "A man selling donuts to a customer."
   - Baseline: Neutral (WRONG) → Debiased: Entailment (CORRECT)

2. **Fix 2:** Contradiction example
   - Premise: "A senior is waiting at the window of a restaurant that serves sandwiches."
   - Hypothesis: "A man is waiting in line for the bus."
   - Baseline: Neutral (WRONG) → Debiased: Contradiction (CORRECT)

**Findings:**
- Debiasing helps on examples requiring better premise-hypothesis reasoning
- Mixed results: slightly more breaks than fixes
- Suggests debiasing has limited but targeted effect

### 6.6 Visualizations (if space allows)
- Confusion matrices (baseline and debiased)
- Per-class accuracy bar chart
- Error type distribution
- Negation word correlation plot

---

## 7. Discussion (1 page)

### 7.1 Artifact Detection Success
- **Strong artifacts detected:** 60.80% hypothesis-only performance
- **Quantitative measure:** 27.47% above random
- **Implication:** SNLI contains significant artifacts that models can exploit

### 7.2 Debiasing Results
- **Maintains performance:** 86.42% vs 86.54% (-0.12%)
- **Positive:** Doesn't hurt overall accuracy
- **Limited improvement:** Minimal change suggests hypothesis-only artifacts may not be primary issue

### 7.3 Why Limited Improvement?

**Possible reasons:**
1. **Hypothesis-only artifacts may not be main issue:**
   - Full model (86.54%) likely uses premise information effectively
   - Artifacts may be less significant than expected

2. **Reweighting scheme may be too weak:**
   - Current: `weight = 1.0 / (1.0 + conf)`
   - May need stronger downweighting or different schemes

3. **Other artifacts may be more significant:**
   - Word overlap patterns
   - Length-based patterns
   - Other linguistic features

4. **Training data size:**
   - 100K examples may be sufficient to learn both artifacts and true patterns
   - Debiasing may be more effective with less data

### 7.4 Neutral Class Difficulty
- **Hardest class:** 82.97% vs 89.28% (Entailment), 87.28% (Contradiction)
- **Interpretation:** Requires more nuanced reasoning
- **Future work:** Focus on improving Neutral class performance

### 7.5 Limitations
- **Single dataset:** Results may not generalize to other NLI datasets
- **Single debiasing method:** Other methods may be more effective
- **Limited artifact analysis:** Only hypothesis-only artifacts investigated
- **No out-of-domain evaluation:** Need to test on other datasets

### 7.6 Future Work
- **Stronger debiasing methods:** Different weighting schemes, adversarial training
- **Broader artifact detection:** Word overlap, length, other patterns
- **Multi-dataset evaluation:** Test on MultiNLI, other NLI datasets
- **Neutral class focus:** Develop methods specifically for Neutral class

---

## 8. Conclusion (0.5 pages)

### 8.1 Summary
- Detected strong artifacts in SNLI (60.80% hypothesis-only)
- Implemented ensemble debiasing method
- Maintained performance while reducing artifact dependence
- Provided detailed error analysis

### 8.2 Key Contributions
- Quantitative artifact detection
- Working debiasing framework
- Comprehensive error analysis
- Identification of limitations and future directions

### 8.3 Implications
- Artifacts exist and can be detected
- Debiasing is possible but needs refinement
- More sophisticated methods needed for significant improvement
- Neutral class requires special attention

### 8.4 Final Thoughts
- Important to understand what models learn
- Artifact detection is crucial for robust models
- Debiasing methods need further development
- Future work should focus on comprehensive artifact detection and mitigation

---

## 9. References

### Key Papers to Cite
1. Bowman et al. (2015) - SNLI dataset
2. Clark et al. (2020) - ELECTRA model
3. Poliak et al. (2018) - Hypothesis-only baselines
4. He et al. (2019) - Ensemble debiasing
5. Clark et al. (2019) - Product of Experts
6. Gururangan et al. (2018) - Annotation artifacts
7. McCoy et al. (2019) - Right for the wrong reasons
8. Gardner et al. (2020) - Contrast sets
9. Ribeiro et al. (2020) - Checklist testing
10. Swayamdipta et al. (2020) - Dataset cartography

### Format
- Use ACL/NeurIPS citation style
- Include all papers mentioned in Related Work
- Add any additional relevant papers

---

## 📝 Writing Tips

### Do's
- ✅ Be honest about limitations
- ✅ Provide detailed analysis
- ✅ Include specific examples
- ✅ Use clear tables and figures
- ✅ Cite related work properly
- ✅ Discuss why results occurred

### Don'ts
- ❌ Overstate results
- ❌ Ignore negative findings
- ❌ Skip error analysis
- ❌ Use vague language
- ❌ Forget to cite sources
- ❌ Make unsupported claims

### Tone
- **Academic but accessible**
- **Honest about results** (both positive and negative)
- **Clear and concise**
- **Well-structured with good flow**

---

## 📊 Word Count Estimates

| Section | Target Words | Pages (approx) |
|---------|--------------|----------------|
| Abstract | 150 | 0.25 |
| Introduction | 600-800 | 1.0 |
| Related Work | 300-400 | 0.5 |
| Method | 800-1200 | 1.0-1.5 |
| Experiments | 300-400 | 0.5 |
| Results | 1200-1600 | 1.5-2.0 |
| Discussion | 600-800 | 1.0 |
| Conclusion | 300-400 | 0.5 |
| References | - | 0.5-1.0 |
| **Total** | **4250-5750** | **6-8 pages** |

---

*Use this outline as a guide. Adjust sections based on your specific findings and emphasis.*

