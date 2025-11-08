# Project Requirements: Analyzing and Mitigating Dataset Artifacts

## 📋 Project Overview

**Goal:** Investigate whether pre-trained models truly "solve" tasks or exploit spurious correlations (dataset artifacts)

**Problem:** Models can achieve high performance by learning shortcuts rather than understanding the task, leading to failures when artifacts are not present

---

## 🎯 Core Tasks

### Part 1: Analysis (≥1 page in paper)

**Objective:** Train a model and analyze its performance and shortcomings

**Requirements:**
1. Train a model on one of the recommended datasets
2. Conduct analysis to identify dataset artifacts
3. Provide specific examples of errors
4. Characterize general classes of mistakes with visualizations (charts, graphs, tables)

**Recommended Datasets:**
- Stanford NLI (SNLI) - Bowman et al., 2015
- MultiNLI - Williams et al., 2018
- SQuAD - Rajpurkar et al., 2016
- HotpotQA - Yang et al., 2018

**Recommended Model:**
- ELECTRA-small (Clark et al., 2020)
- Alternative: Any model you prefer

**Analysis Methods (choose 1-3):**

| Category | Method | Description |
|----------|--------|-------------|
| **Changing Data** | Contrast sets | Use existing or hand-design small annotated examples |
| | Checklist sets | Systematic behavioral testing |
| | Adversarial challenge sets | Use existing adversarial examples |
| **Changing Model** | **Model ablations** | **Hypothesis-only NLI, sentence-factored models** |
| **Statistical Test** | Competency problems | Find spurious n-gram correlations |

---

### Part 2: Fix It

**Objective:** Improve the issues identified in Part 1

**Requirements:**
1. Pick a method to address the artifacts
2. Implement and evaluate the fix
3. Provide deep analysis beyond just accuracy numbers
4. Show examples of predictions that are fixed
5. Discuss breadth of fix and any tradeoffs

**Recommended Methods:**

| Method | Description | References |
|--------|-------------|------------|
| **Ensemble-based debiasing** | **Train weak model to learn artifacts, then train main model on residual** | **He et al., 2019; Clark et al., 2019** |
| Dataset cartography | Focus on hard/ambiguous examples | Swayamdipta et al., 2020 |
| Adversarial training | Train on challenge sets or augmented data | Liu et al., 2019; Zhou & Bansal, 2020 |
| Contrastive training | Use contrastive learning | Dua et al., 2021 |

**Note:** You can use open-source repositories associated with these papers

---

## 📊 Evaluation Expectations

### What to Report

**Baseline Results:**
- Accuracy of initial trained model
- Error analysis with examples
- Identification of artifact patterns

**Fix Results:**
- Overall accuracy change
- Performance on targeted error subsets
- Examples of fixed predictions
- Analysis of why fixes worked/didn't work

**Critical Questions to Address:**
- How effective is your fix?
- Did you address the targeted errors?
- How broad was the fix?
- Did overall performance improve/decline?
- What tradeoffs exist?

### Success Criteria

✅ **Your fix doesn't need to work perfectly!**
- Even negative results are acceptable with good analysis
- Small improvements on hard subsets count (e.g., 10% hardest examples)
- Important: Show you can analyze WHY things did/didn't work

---

## 📝 Deliverables

### 1. Code (Documentary purposes only)
- Submit any code you wrote
- Not graded on execution
- Do not include large data files

### 2. Final Paper (Primary deliverable)

**Format:** ACL/NeurIPS conference style

**Length:** 3-8 pages (excluding references)
- Single person: Closer to 3-5 pages
- Two person team: 5-8 pages with more depth

**Required Sections:**
1. **Abstract** - Summary of motivation, methodology, results
2. **Introduction** - Problem, approach, contributions
3. **Related Work** - Cite relevant papers
4. **Approach/Method** - Technical details of your analysis and fix
5. **Experiments** - Dataset, models, evaluation setup
6. **Results** - Tables, graphs, analysis
7. **Discussion** - What worked, what didn't, why
8. **Conclusion** - Summary and future work
9. **References**

---

## 📊 Grading Rubric (100 points)

| Category | Points | Criteria |
|----------|--------|----------|
| **Scope** | 25 | Sufficient depth; not just shallow analysis of base system |
| **Implementation** | 30 | Technically sound approach; correct methodology |
| **Results/Analysis** | 30 | Deep analysis with examples; explain successes/failures; include baselines and ablations |
| **Clarity/Writing** | 15 | Clear paper structure; good abstract/intro/method/results sections |

### Key Grading Notes

**Scope (25 pts):**
- Must show depth beyond basic system analysis
- Should demonstrate understanding of artifacts

**Implementation (30 pts):**
- Technically correct approach
- Proper connection between method and goal
- No major technical errors

**Results/Analysis (30 pts):**
- Must include:
  - Baseline results
  - Best method results
  - Ablations (minimal changes to assess contribution)
- Show examples, visualizations, patterns
- Explain WHY things worked/didn't work

**Clarity/Writing (15 pts):**
- Clear motivation and hypothesis
- Well-described methodology
- Clear graphs and tables (not just inline numbers)
- Proper related work citations

---

## 💡 Scope Guidelines

### Code Expectations

**Important:** You may not write much code!
- Great projects might modify only 20 lines
- Most work is in: (a) studying data, (b) understanding modifications, (c) analyzing results

### Single Person vs. Two Person Projects

**Single Person:**
- Stick closer to settings in prior work
- Try straightforward modification methods
- Same basic analyses but less detailed

**Two Person Team:**
- Try more sophisticated modifications
- Could try two different approaches to same problem
- More in-depth analysis expected

---

## ⚡ Computational Resources

### Training Time Expectations

**Budget:** 5-15 hours for initial model training

**Tips:**
- Debug on small amounts of data first
- Use checkpoints to evaluate needed training time
- Start runs, evaluate after 2 hours to assess if sufficient
- Most middle experiments shouldn't need hours of training

**Recommendations:**
- Use ELECTRA-small (computationally easier than larger models)
- Limit training data if needed for faster iterations
- Use HuggingFace checkpointing for efficiency

---

## 🎯 Success Strategy

### Structure Work in Phases

Even if everything doesn't work:
- ✅ Phase 1: Get something working
- ✅ Phase 2: Run some experiments
- ✅ Phase 3: Get results to report

Avoid all-or-nothing outcomes!

### Analysis Before Implementation

**Critical:** Plan evaluation BEFORE diving into Part 2
- What improvements do you expect?
- What criteria will you evaluate?
- If impact seems very limited, try something different
- OR focus on small but important class of examples

### If Things Don't Work

Must still show:
1. ✅ Correctly implemented what you intended
2. ✅ Can analyze results to understand why things went wrong
3. ✅ Have meaningful results/analysis to report

**NOT acceptable:** "I tried X and wrote 100 lines but it still crashes"

---

## 📚 Key References

### Dataset Papers
- Bowman et al., 2015 - SNLI dataset
- Williams et al., 2018 - MultiNLI
- Rajpurkar et al., 2016 - SQuAD
- Yang et al., 2018 - HotpotQA

### Analysis Methods
- Poliak et al., 2018 - Hypothesis-only baselines
- Gardner et al., 2020 - Contrast sets
- Ribeiro et al., 2020 - Checklist
- Gardner et al., 2021 - Competency problems

### Debiasing Methods
- He et al., 2019 - Ensemble debiasing
- Clark et al., 2019 - Product of Experts
- Swayamdipta et al., 2020 - Dataset cartography
- Zhou & Bansal, 2020 - Adversarial methods
- Utama et al., 2020 - Debiasing techniques

### Models
- Clark et al., 2020 - ELECTRA

---

## ✅ Project Checklist

### Part 1: Analysis
- [ ] Train baseline model on chosen dataset
- [ ] Run error analysis
- [ ] Identify artifact patterns
- [ ] Create visualizations (charts/tables)
- [ ] Collect specific error examples
- [ ] Write 1+ page analysis

### Part 2: Fix
- [ ] Choose debiasing method
- [ ] Implement fix
- [ ] Evaluate on same test set as baseline
- [ ] Collect improvement examples
- [ ] Run ablations if applicable
- [ ] Analyze why fix worked/didn't work

### Paper Writing
- [ ] Abstract (motivation, method, results)
- [ ] Introduction (problem, approach, contributions)
- [ ] Related work (cite key papers)
- [ ] Method section (technical details)
- [ ] Experiments section (setup)
- [ ] Results section (tables, graphs, analysis)
- [ ] Discussion (interpretation, limitations)
- [ ] Conclusion (summary, future work)
- [ ] References (proper citations)

### Submission
- [ ] Paper: 3-8 pages (excluding references)
- [ ] Code files (no large data files)
- [ ] All visualizations/tables included
- [ ] Proper ACL formatting

---

## 🎓 Example Project Flow

**Scenario:** Following up on Dataset Cartography (Swayamdipta et al., 2020)

**If using their repository:**
- Apply to different dataset (e.g., SQuAD instead of their dataset)

**Questions to explore:**
1. Split dataset into easy/hard/ambiguous - what do subsets share?
2. What makes examples hard vs. easy?
3. What role does each subset play in training?
4. Can we make model "pay more attention" to hard/ambiguous examples?
5. Try approaches beyond their work (data augmentation, soft reweighting)

---

## 📞 Getting Help

**Resources:**
- Starter code: https://github.com/gregdurrett/fp-dataset-artifacts
- HuggingFace transformers documentation
- Example papers cited in spec
- Course staff (check with them ~1-2 weeks before deadline)

**Timeline Advice:**
- Have preliminary results 1-2 weeks before deadline
- Ensure you can show analysis even if results aren't perfect
- Structure work to avoid "nothing works" scenario

---

*Document created: November 8, 2024*
*Source: CS388 Final Project Specification*

