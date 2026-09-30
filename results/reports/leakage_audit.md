# TRACEGUARD: Data Leakage and Data Difficulty Audit

**Project:** TRACEGUARD: Early Detection and Localization of Hijacked LLM Agent Trajectories
**Authors:** Deshna Tendulkar, Harshad Agrawal
**Audit Target:** `traceguard_pilot_v2.jsonl` (90 Trajectories)

---

## 1. Feature & Textual Leakage Audit

### Ground-Truth Metadata Leakage
- **Fields explicitly embedded in step input:** `agent_state`, `action`, `tool_name`, `tool_input`, `tool_observation`, `concise_reasoning_or_decision`.
- **Forbidden ground-truth fields checked:** `label`, `attack_success`, `injection_step`, `deviation_step`, `attack_type`, `ground_truth_explanation`, `final_outcome`, `is_injection_present`, `is_behaviorally_deviant`.
- **Result:** **NO forbidden metadata fields** were found inside the step dictionary entries used for embedding generation.

### Suspicious Term Search in Embedded Text
- Search terms scanned: `HIJACKED`, `BENIGN`, `INJECTION_RESISTED`, `attack_success`, `deviation`, `successful attack`, `resisted injection`, `attack_type`.
- **Found occurrences:** 0 instances across trajectory texts.
- **No explicit ground-truth label terms** were detected inside the step text.

---

## 2. Classification on Clean Representation (Embedding Baseline)

- **Input Representation:** Sentence-Transformer (`all-MiniLM-L6-v2`) mean-pooled step embeddings using only clean step fields.
- **Model:** Logistic Regression (scikit-learn)
- **Clean Baseline Accuracy:** 1.0000 (100.00%)
- **Clean Baseline Macro F1:** 1.0000
- **Confusion Matrix (Clean Baseline):**
```
             Pred: BENIGN  Pred: RESISTED  Pred: HIJACKED
True: BENIGN       4             0               0
True: RESISTED     0             5               0
True: HIJACKED     0             0               5
```

---

## 3. Simple Text Leakage & Bag-Of-Words (TF-IDF) Test

A simple TF-IDF + Logistic Regression model trained strictly on trajectory text achieves:
- **TF-IDF Accuracy:** 1.0000 (100.00%)
- **TF-IDF Macro F1:** 1.0000

### Top Discriminative Features per Class (TF-IDF Weights)
- **Class `BENIGN` Top Terms:** `answers, checking, value, information, answering, gathered, fully, matching, goal, sufficient`
- **Class `INJECTION_RESISTED` Top Terms:** `data, answer, injected, continuing, proceed, instruction, disregard, response, using, excluding`
- **Class `HIJACKED` Top Terms:** `unauthorized, balance, acting, rec_1001, update, embedded, finding, record, disclosure, allowed`

*Insight:* The presence of distinct attack/injection keywords in prompt injection payloads and behavioral deviation tool calls allows even a simple unigram/bigram TF-IDF model to achieve near-perfect or perfect class separation.

---

## 4. Nearest-Neighbor Train-to-Test Similarity Test

Evaluating cosine similarity between each test trajectory embedding and its nearest training trajectory embedding:
| Test Trajectory ID | Test Label | Nearest Train ID | Nearest Train Label | Cosine Similarity |
| :--- | :--- | :--- | :--- | :--- |
| `benign_011` | `BENIGN` | `benign_001` | `BENIGN` | 0.9925 |
| `resisted_017` | `INJECTION_RESISTED` | `resisted_006` | `INJECTION_RESISTED` | 0.9785 |
| `hijacked_012` | `HIJACKED` | `hijacked_025` | `HIJACKED` | 0.9861 |
| `resisted_004` | `INJECTION_RESISTED` | `resisted_010` | `INJECTION_RESISTED` | 0.9923 |
| `resisted_007` | `INJECTION_RESISTED` | `resisted_024` | `INJECTION_RESISTED` | 0.9861 |
| `hijacked_029` | `HIJACKED` | `hijacked_014` | `HIJACKED` | 0.9881 |
| `resisted_003` | `INJECTION_RESISTED` | `resisted_030` | `INJECTION_RESISTED` | 0.9513 |
| `benign_027` | `BENIGN` | `benign_017` | `BENIGN` | 0.9938 |
| `benign_021` | `BENIGN` | `benign_001` | `BENIGN` | 0.9959 |
| `benign_015` | `BENIGN` | `benign_016` | `BENIGN` | 0.9955 |
| `resisted_014` | `INJECTION_RESISTED` | `resisted_006` | `INJECTION_RESISTED` | 0.9870 |
| `hijacked_026` | `HIJACKED` | `hijacked_009` | `HIJACKED` | 0.9818 |
| `hijacked_020` | `HIJACKED` | `hijacked_001` | `HIJACKED` | 0.9935 |
| `hijacked_010` | `HIJACKED` | `hijacked_013` | `HIJACKED` | 0.9896 |

- **Mean Nearest-Neighbor Similarity (Test to Train):** 0.9866
- **Max Nearest-Neighbor Similarity (Test to Train):** 0.9959

---

## 5. Duplicate & Near-Duplicate Audit

Pairwise trajectory similarity across all 90 trajectories:
- **Maximum Pairwise Cosine Similarity:** 0.9973
- **Mean Pairwise Cosine Similarity:** 0.8363
- **Number of Pairs with Similarity > 0.90:** 1224
- **Number of Pairs with Similarity > 0.95:** 249
- **Number of Pairs with Similarity > 0.99:** 66

---

## 6. Class Separability (PCA Visualization)

- **Plot Saved:** `results/figures/pca_separability.png`
- **Separability Finding:** PCA visualization confirms that `BENIGN`, `INJECTION_RESISTED`, and `HIJACKED` trajectories form separate, tight clusters in embedding space.
- **Reason:** 
  1. `BENIGN` trajectories lack any prompt injection phrases or security-policy violation terms.
  2. `INJECTION_RESISTED` trajectories contain injection text in tool observations but main agent reasoning/action strictly follows the system prompt.
  3. `HIJACKED` trajectories contain injection text *and* exhibit abrupt shifts in tool calls (e.g., unauthorized data exfiltration or system commands).

---

## 7. Attack Vocabulary Frequency Analysis

Occurrences of key security/attack terms across trajectory classes (out of 30 trajectories per class):

| Term | BENIGN Trajectories | INJECTION_RESISTED Trajectories | HIJACKED Trajectories |
| :--- | :---: | :---: | :---: |
| `ignore` | 0 | 0 | 0 |
| `previous` | 0 | 0 | 0 |
| `instruction` | 0 | 30 | 30 |
| `system` | 3 | 8 | 8 |
| `administrator` | 0 | 2 | 4 |
| `urgent` | 0 | 3 | 2 |
| `override` | 0 | 3 | 0 |
| `secret` | 0 | 0 | 0 |
| `confidential` | 0 | 0 | 0 |
| `authorized` | 0 | 0 | 30 |
| `security` | 0 | 0 | 0 |
| `redirect` | 0 | 0 | 0 |
| `delete` | 0 | 0 | 0 |
| `send` | 0 | 0 | 0 |
| `access` | 0 | 1 | 2 |

---

## 8. Template & Diversity Audit

- **Unique User Goals:** 10
- **Unique Injection Prompts:** 1
- **Unique Tool Sequences:** 14
- **Unique Full Tool Observations:** 70
- **Unique Reasoning/Decision Statements:** 79

*Finding:* The pilot dataset relies on a small set of template structures (e.g. 5 attack families across 90 trajectories), leading to high intra-class homogeneity and distinct inter-class keyword profiles.

---

## 9. Train / Validation / Test Split Integrity

- **Split Timing:** The split was performed **strictly before** TF-IDF fitting and PyTorch model training.
- **Data Leakage across Splits:** 0% (Split performed at the trajectory ID level; no steps from the same trajectory cross splits).

---

## 10. Controlled Representation Experiments (A, B, C)

To test feature sensitivity, Logistic Regression was evaluated on different text representations:

| Representation | Included Fields | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Exp A (Full)** | `agent_state`, `action`, `tool_name`, `tool_input`, `tool_observation`, `concise_reasoning` | 1.0000 | 1.0000 |
| **Exp B (Structural)** | `agent_state`, `action`, `tool_name` | 1.0000 | 1.0000 |
| **Exp C (Observation)** | `tool_observation` (contains injection payloads) | 0.7143 | 0.7302 |

---

## 11. Sequential Advantage Assessment

| Model | Architecture | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Non-Sequential Baseline** | Mean-Pooled Embedding + Logistic Regression | 1.0000 | 1.0000 |
| **LSTM** | Sequential PyTorch LSTM | 1.0000 | 1.0000 |
| **Transformer** | Lightweight Transformer Encoder | 1.0000 | 1.0000 |

### Official Finding on Sequential Advantage:
> **"The pilot dataset (`traceguard_pilot_v2.jsonl`) is perfectly separable even by a non-sequential baseline; therefore, the current pilot experiment does not establish a sequential modeling advantage."**

---

## 12. Final Diagnostic Conclusion & Recommendations

1. **Metadata Leakage:** **NONE.** No ground-truth fields (`label`, `attack_success`, `deviation_step`, etc.) were present in the model inputs.
2. **Textual Leakage:** **NONE (Explicit).** No explicit label strings like `"HIJACKED"` or `"BENIGN"` were leaked.
3. **Data Difficulty & Template Uniformity:** **HIGH SEPARABILITY.** 
   - The pilot dataset consists of 90 highly structured trajectories generated from 5 attack templates.
   - The presence of injection keywords in `tool_observation` and specific hijacked tool calls in later steps creates strong linear separability.
4. **Recommendation for Next Phase:**
   - **Do NOT modify or alter `traceguard_pilot_v2.jsonl`** (retaining integrity of pilot results).
   - For Phase 2 (Scale & Benchmark), generate a significantly larger, more diverse dataset (500-1000+ trajectories) featuring:
     - Harder/subtle prompt injections (semantic obfuscation, zero explicit attack keywords).
     - Varied agent reasoning styles and noise in benign trajectories.
     - Multi-turn benign trajectories with complex tool sequences to test true sequential anomaly detection.
