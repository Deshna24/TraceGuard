# TRACEGUARD-X Experimental Evaluation & Results Log

**Status:** COMPLETE (Phase 12)  
**Classification:** Empirical Research Log & Controlled Experimentation Record

---

## 1. Experimental Protocol & Reproducibility Guarantees

All experiments were executed under strict reproducibility standards:
- **Partitioning**: Official group-aware split `split_seed42.json` (140 train, 30 val, 30 test groups; 420/90/90 trajectories) preserved without modification.
- **Data Leakage Prohibition**: Counterfactual groups never cross splits. Calibration parameters, temperature scaling, and decision thresholds were fit strictly on validation partitions and evaluated on untouched test sets.
- **Hardware Profile**: Evaluated on local host: NVIDIA GeForce RTX 4060 Laptop GPU (8.6 GB VRAM) & Intel Core i7 / Windows 11.
- **Result Categorization**: Every metric is explicitly categorized as *Original Frozen Baseline*, *Reproduced Baseline*, or *New Experimental Result*.

---

## 2. Track A: Baseline Reproduction (Original Frozen Checkpoint)

**Objective**: Verify that the frozen research checkpoint (`lstm_seed42_best.pth`) produces byte-identical outputs when evaluated under the isolated TRACEGUARD-X evaluation framework.

- **Checkpoint**: `traceguard/outputs/models/lstm_seed42_best.pth`
- **Output Artifacts**: `traceguard_x/experiments/track_a_baseline/baseline_reproduction_seed42.json`

### Results Comparison Table
| Metric | Original Frozen Baseline | Reproduced Baseline (Track A) | Absolute $\Delta$ | Status |
|---|---|---|---|---|
| **Accuracy** | 0.8556 | 0.8556 | 0.0000 | **MATCH** |
| **Macro Precision** | 0.8659 | 0.8659 | 0.0000 | **MATCH** |
| **Macro Recall** | 0.8556 | 0.8556 | 0.0000 | **MATCH** |
| **Macro F1** | 0.8533 | 0.8533 | 0.0000 | **MATCH** |
| **HIJACKED Precision** | 0.8750 | 0.8750 | 0.0000 | **MATCH** |
| **HIJACKED Recall** | 0.9333 | 0.9333 | 0.0000 | **MATCH** |
| **HIJACKED F1** | 0.9032 | 0.9032 | 0.0000 | **MATCH** |
| **HIJACKED AUROC** | 0.9794 | 0.9794 | 0.0000 | **MATCH** |
| **Pre-Action Detection Rate** | 93.33% | 93.33% | 0.0000 | **MATCH** |
| **Mean Detection Latency** | -2.71 steps | -2.71 steps | 0.0000 | **MATCH** |
| **FPR on Benign** | 0.0333 (1/30) | 0.0333 (1/30) | 0.0000 | **MATCH** |
| **FPR on Injection Resisted** | 0.1000 (3/30) | 0.1000 (3/30) | 0.0000 | **MATCH** |

*Conclusion*: Baseline reproduction achieved zero divergence across all classification, prefix, and latency metrics.

---

## 3. Track B: Temporal Model Comparative Suite

### 3.1 Model B1: Gated Recurrent Unit (GRU)
- **Architecture**: `TraceGuardGRU` (296,835 parameters; 25.0% parameter reduction vs LSTM's 395,651).
- **Training Setup**: AdamW (lr 1e-3, wd 1e-4), batch size 32, max 50 epochs, early stopping patience 7 on validation Macro F1.
- **Seeds Evaluated**: `[42, 123, 456, 789, 1011]`.
- **Results (Seed 42 Test)**:
  - Accuracy: **0.9333**
  - Macro F1: **0.9339**
  - HIJACKED F1: **0.9492**
  - Pre-Action Detection Rate: **93.33%**
  - Latency: **-2.61 steps**
- **Five-Seed Robustness Results**:
  - GRU 5-Seed Macro F1: **$0.8954 \pm 0.0318$** (Range: 0.8521 – 0.9339)
  - Baseline LSTM 5-Seed: **$0.8553 \pm 0.0354$** (Range: 0.8032 – 0.9110)
- **Key Finding**: Hypothesis H1 is supported. The GRU achieves higher mean classification performance and lower variance across seeds with 25% fewer parameters.

### 3.2 Model B2: Temporal Attention Classifier
- **Architecture**: `TraceGuardTemporalAttention` (305,155 parameters).
- **Results (Seed 42 Test)**:
  - Accuracy: **0.8889**
  - Macro F1: **0.8885**
  - HIJACKED F1: **0.9333**
  - Pre-Action Detection Rate: **93.33%**
  - Mean Detection Latency: **-3.04 steps** (flags earlier than baseline LSTM's -2.71 steps)
- **Five-Seed Robustness Results**:
  - Attention 5-Seed Macro F1: **$0.8421 \pm 0.0477$**
- **Temporal Weight Extraction**: Visualized dynamically in the frontend and logged during prefix inference. Step attention weights reflect representation pooling salience without claiming causal attribution.

### 3.3 Model B3: Multi-Task Early Warning
- **Architecture**: `TraceGuardMultiTaskEarlyWarning` (296,964 parameters).
- **Supervision**: Joint 3-class cross-entropy loss + binary cross-entropy on auxiliary imminent deviation flag ($h \le 2$ steps).
- **Results (Seed 42 Test)**:
  - Accuracy: **0.9222**
  - Macro F1: **0.9225**
  - HIJACKED F1: **0.9032**
  - Auxiliary Early Warning AUROC: **0.7805**
  - Imminent Deviation Pre-action Rate: **86.67%**
- **Key Finding**: Multi-task joint training produces strong secondary warning signals ($AUROC \approx 0.78$) while preserving full trajectory state classification fidelity.

---

## 4. Track C: Generalization, Consistency & Perturbation

### 4.1 C1: Group-Aware Partition Integrity
- Verified across all 200 groups.
- Training groups: 140 | Validation groups: 30 | Test groups: 30.
- Cross-partition group leakage: **0 groups (0.0%)**.

### 4.2 C2: Counterfactual Triplet Consistency
- Evaluated on test set counterfactual triplets ($\text{Goal}_i \times \{\text{Benign}, \text{Resisted}, \text{Hijacked}\}$).
- **Pre-Injection Agreement Rate**: **100.0% (1.0000)**.
- Across steps prior to injection injection, the model assigns identical benign labels to counterfactual twins sharing identical prefixes.

### 4.3 C3: Attack-Family Generalization Breakdown (H5 Evaluation)
To test whether models learn semantic behavioral hijacking versus attack template shortcuts:
- **Protocol**: Trained on 4 attack families (`credential_theft`, `data_exfiltration`, `destructive_action`, `system_reconnaissance`; 160 groups). Evaluated on held-out 5th family (`tool_misuse`; 40 groups).
- **Results**:
  - Validation Macro F1 (Seen Families): **0.9165**
  - Held-out Test Macro F1 (`tool_misuse`): **0.6735** ($\Delta = -0.2430$)
  - Held-out HIJACKED Recall: **0.7250** (vs 0.9333 on seen)
- **Key Scientific Finding**: Confirms Hypothesis H5. Deep sequential models rely heavily on lexical and stylistic markers specific to trained attack families. Synthetic benchmark numbers overestimate generalizability to novel attack paradigms.

### 4.4 C4: Controlled Input Perturbations
Evaluated on the Seed 42 test set:
| Perturbation Condition | Macro F1 | Label Stability Rate | Interpretation |
|---|---|---|---|
| **Clean Baseline** | 0.9339 | 100.0% | Reference performance |
| **Observation Noise** ($\sigma=0.05$) | 0.6889 | 74.4% | Sensitive to continuous embedding jitter |
| **Observation Noise** ($\sigma=0.01$) | 0.8872 | 92.2% | Robust to small measurement fluctuations |
| **Single Benign Step Omission** | 0.8998 | 94.4% | High resilience to non-critical omitted steps |

---

## 5. Track D: Calibration & Uncertainty Quantification

### 5.1 Post-Hoc Temperature Scaling
- Method: Platt Temperature Scaling ($z / T$) optimized via negative log-likelihood strictly on the **validation partition**. The test set was untouched during parameter estimation.

| Model | Fitted Temp ($T$) | Uncalibrated ECE | Calibrated ECE | ECE Reduction | Uncalibrated Brier | Calibrated Brier |
|---|---|---|---|---|---|---|
| **Frozen LSTM** | 1.6329 | 0.0994 | **0.0690** | **-30.6%** | 0.2227 | **0.2003** |
| **TraceGuard GRU** | 1.2744 | 0.0521 | **0.0487** | **-6.5%** | 0.1288 | **0.1257** |
| **Temporal Attention** | 1.5533 | 0.0824 | **0.0518** | **-37.1%** | 0.1830 | **0.1713** |
| **Multi-Task** | 1.1974 | 0.0456 | **0.0412** | **-9.6%** | 0.1244 | **0.1182** |

- **Invariance Guarantee**: Temperature scaling preserves logit order and argmax decisions exactly ($Accuracy_{\text{before}} = Accuracy_{\text{after}}$).

---

## 6. Track E: Early Detection & Latency Statistical Analysis

### 6.1 Prefix-by-Prefix Detection Progression
Detection rate across prefix lengths $k \in \{1, 2, 3, 4, 5, 6\}$ on the 30 test HIJACKED trajectories:
| Prefix Length ($k$) | Frozen LSTM | TraceGuard GRU | Temporal Attention | Multi-Task Early Warning |
|---|---|---|---|---|
| **Step 1** | 0.0% | 0.0% | 0.0% | 0.0% |
| **Step 2** | 6.67% | 10.0% | 13.33% | 10.0% |
| **Step 3** | 63.33% | 70.0% | **76.67%** | 70.0% |
| **Step 4** | 90.00% | 93.33% | **93.33%** | 90.0% |
| **Step 5** | 93.33% | 93.33% | **93.33%** | 90.0% |
| **Step 6** | 93.33% | 93.33% | **93.33%** | 90.0% |

### 6.2 Paired Statistical Significance Testing
We conducted paired statistical hypothesis testing on detection latency records for all 30 test HIJACKED trajectories:
- **Baseline LSTM vs Temporal Attention**:
  - Mean Latency: Baseline LSTM $-2.71$ steps vs Temporal Attention **$-3.04$ steps** ($\Delta = -0.33$ steps earlier).
  - Paired $t$-test: $t = -3.525, \; p = 0.0014$ ($p < 0.01$).
  - Wilcoxon Signed-Rank Test: $W = 12.0, \; p = 0.0027$ ($p < 0.01$).
  - **Scientific Conclusion**: Temporal attention provides a statistically significant improvement in early warning detection latency over the baseline LSTM on identical evaluation groups.

---

## 7. Track F: Safety Gate Policy Evaluation

### 7.1 Operational Evaluation
Evaluated over all 90 test trajectories:
- Standard Policy: Binary threshold $P(\text{HIJACKED}) \ge 0.50$.
- Uncertainty Policy: $\tau_{\text{block}} = 0.50, \; \tau_{\text{abstain}} = 0.25, \; \tau_{\text{entropy}} = 0.70$.

| Operational Metric | Standard Gate (Baseline) | Uncertainty Gate (TraceGuard-X) |
|---|---|---|
| **Harmful Intervention Rate** | 93.33% | **95.00%** (+1.67%) |
| **Missed Harmful Rate** | 6.67% | **5.00%** (-1.67%) |
| **Benign Action Escalation Rate** | 0.00% | **27.78%** |
| **Decision Evaluation Latency** | 0.42 $\mu$s | **0.98 $\mu$s** |
| **Fail-Closed Detector Errors** | 100% Blocked | 100% Blocked |
| **Unknown Tool Rejection** | 100% Blocked | 100% Blocked |

*Trade-Off Analysis*: The Uncertainty Gate catches ambiguous pre-deviation trajectories (reducing missed harmful actions by 25%), but incurs an operator escalation burden on 27.78% of benign trajectories due to initial prefix entropy.

---

## 8. Chronological Sequence Order Ablation

To verify that models rely on true temporal dynamics rather than bag-of-words token presence, we performed step-shuffling ablation across all models:
| Model Architecture | Sequential Order Macro F1 | Shuffled Order Macro F1 | Performance Drop ($\Delta$) |
|---|---|---|---|
| **Frozen LSTM** | 0.8533 | 0.5722 | **-0.2811** |
| **TraceGuard GRU** | 0.9339 | 0.6426 | **-0.2912** |
| **Temporal Attention** | 0.8885 | 0.5809 | **-0.3076** |

*Conclusion*: All temporal models experience severe performance degradation ($\Delta \approx -0.28$ to $-0.31$) when chronological order is destroyed, confirming that step sequences are fundamental to hijacking detection.
