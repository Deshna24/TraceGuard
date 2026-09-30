# TRACEGUARD v3: Temporal Separability Audit Report

**Project:** TRACEGUARD: Early Detection and Localization of Hijacked LLM Agent Trajectories  
**Authors:** Deshna Tendulkar, Harshad Agrawal  
**Dataset:** `traceguard_v3.jsonl` (300 Trajectories, 100 Counterfactual Groups)  
**Focus:** `INJECTION_RESISTED` vs `HIJACKED` Temporal Separability  

---

## 1. Purpose & Executive Summary

The purpose of this audit is to resolve **WHY** TF-IDF achieves ~98% accuracy when classifying `INJECTION_RESISTED` vs `HIJACKED` trajectories in TRACEGUARD v3. Specifically, we investigate whether this high performance is driven by **POST-DEVIATION** behavioral evidence (expected and required for early detection) or if the classes are trivially separable **BEFORE** the behavioral deviation step due to dataset artifact shortcuts.

### Key Audit Conclusions:
- **Pre-Deviation Accuracy:** **50.00%** (Exact chance level = 50.00%).
- **Pre-Injection & Through-Injection Accuracy:** **50.00%** and **50.00%** (Exact chance level = 50.00%).
- **At-Deviation & Full Trajectory Accuracy:** **97.50%** (At-Deviation) and **97.50%** (Full Trajectory).
- **Counterfactual Pre-Deviation Equivalence:** **100 out of 100 counterfactual R/H pairs are 100% textually identical** prior to the deviation step (Mean Cosine Similarity: **1.0000**, Mean Jaccard Token Overlap: **1.0000**).
- **Sanity Check Status:** **PASSED PERFECTLY**. TRACEGUARD v3 contains **ZERO** pre-deviation class shortcuts. High classification accuracy emerges strictly at and after the behavioral deviation step.

---

## 2. Classification Accuracy Across Temporal Representations

Evaluated using **GroupKFold** cross-validation grouped by `counterfactual_group` (5-fold CV, 100 counterfactual groups):

| Representation | Cutoff Definition | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **PRE-INJECTION** | Strictly BEFORE `injection_step` | 0.5000 | 0.3333 |
| **THROUGH-INJECTION** | THROUGH `injection_step` (includes injection observation) | 0.5000 | 0.3333 |
| **PRE-DEVIATION** | Strictly BEFORE `deviation_step` | 0.5000 | 0.3333 |
| **AT-DEVIATION** | THROUGH `deviation_step` (includes first deviant step) | 0.9750 | 0.9746 |
| **FULL TRAJECTORY** | Complete trajectory | 0.9750 | 0.9746 |

> **Note on Macro F1 = 0.3333 for 50.00% Accuracy:** When pre-deviation trajectory texts are 100% identical between `INJECTION_RESISTED` and `HIJACKED` counterfactual twins, the TF-IDF feature matrix contains zero discriminative features. The linear classifier predicts all samples into a single default class, resulting in 50% accuracy, 1.0 recall for class 0, 0.0 recall for class 1, and a macro-averaged F1 of 0.3333.

---

## 3. Exact Counterfactual Pre-Deviation Pairwise Comparison

Comparing the sanitized text of `INJECTION_RESISTED` vs `HIJACKED` twins in each of the 100 counterfactual groups prior to the `deviation_step`:

- **Total Matched Counterfactual R/H Pairs:** 100
- **Identical Pre-Deviation Text Count:** 100 / 100 (**100.0%**)
- **Identical Pre-Deviation Tool Sequence Count:** 100 / 100 (**100.0%**)
- **Mean Cosine Similarity:** **1.0000** (Min: 1.0000, Max: 1.0000)
- **Mean Token Overlap (Jaccard):** **1.0000** (Min: 1.0000, Max: 1.0000)
- **Differing Counterfactual Groups Before Labeled Deviation:** None (0 groups differ before deviation).

---

## 4. Pre-Deviation Class Signal Analysis

TF-IDF feature importance analysis trained strictly on `PRE-DEVIATION` prefix texts:
- **Maximum Absolute Weight/Coefficient:** `0.000000`
- **Top Terms (INJECTION_RESISTED):** `10, 100, 101, 104, 108, 1089, 11, 110, 113, 117`
- **Top Terms (HIJACKED):** `wrap, worth, workflow, wireless, window, week, website, webcam, warranty, warehouse`

**Finding:** Because 100% of pre-deviation trajectory texts are identical across counterfactual pairs, all Logistic Regression coefficients are equal to zero ($w_i \approx 0.0$). There are **NO** pre-deviation class-specific vocabulary signals in v3.

---

## 5. Structural & Metadata Leakage Tests

1. **Length Leakage Test:**
   - Evaluated on features: `[number_of_steps, number_of_tool_calls, number_of_observations, token_count]`
   - **Pre-Deviation Length Features Grouped CV Accuracy:** **50.00%** (Chance level = 50.0%)
   - **Full Trajectory Length Features Grouped CV Accuracy:** **75.00%** (75.0% - HIJACKED trajectories naturally contain 1 additional tool execution step due to the hijacking payload).

2. **Pre-Deviation Tool-Sequence Leakage Test:**
   - Evaluated on: Sequence of `tool_name`s strictly before `deviation_step`.
   - **Grouped CV Accuracy:** **50.00%** (Exact chance level = 50.0%).

---

## 6. Prefix Accuracy Curve

Classification accuracy as a function of the fraction of trajectory observed (Grouped 5-Fold CV):

| Trajectory Fraction Observed | Grouped CV Accuracy |
| :---: | :---: |
| **20%** | 0.5750 (57.5%) |
| **40%** | 0.6300 (63.0%) |
| **60%** | 0.9700 (97.0%) |
| **80%** | 0.9700 (97.0%) |
| **100%** | 0.9750 (97.5%) |

- **Plot Saved:** `results/figures/tfidf_temporal_separability.png`

---

## 7. Answer to Critical Research Question

> **"Is the 98% TF-IDF accuracy caused primarily by POST-DEVIATION behavioral evidence, or can the classes already be trivially separated before behavioral deviation?"**

### Empirically Grounded Answer:
The **98% TF-IDF accuracy is caused EXCLUSIVELY by POST-DEVIATION behavioral evidence**.

1. **Before the deviation step (`PRE-DEVIATION`), classification accuracy is exactly 50.00% (chance level).**
2. **100 out of 100 (100.0%) counterfactual `INJECTION_RESISTED` and `HIJACKED` pairs are 100% textually and structurally identical** up to the deviation step. They share identical user goals, identical tool call histories, identical agent reasoning, and identical injected observation texts.
3. **Accuracy surges from 50.00% to 97.50% AT the deviation step**, precisely when the hijacked agent performs its first unauthorized tool call outside the scope of the original user goal.

---

## 8. Decision & Next Steps

### Audit Decision:
**TRACEGUARD v3 PASSES THE TEMPORAL SEPARABILITY SANITY CHECK.**

The dataset is free of pre-deviation shortcuts, leakage, or counterfactual imbalances. High classification performance reflects genuine behavioral detection at the moment of hijacking deviation.

- **LSTM & Transformer Models:** Safe to proceed with sequential early detection modeling when authorized.
- **Dataset Modification:** No changes required for `traceguard_v3.jsonl`.
