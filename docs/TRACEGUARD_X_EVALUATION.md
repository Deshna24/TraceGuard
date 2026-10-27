# TRACEGUARD-X Comprehensive Evaluation & Synthesis Report

**Status:** COMPLETE (Phase 12)  
**Classification:** Scientific Benchmark Synthesis & Statistical Evaluation

---

## 1. Evaluation Methodology & Metric Definitions

To ensure scientific rigor, TRACEGUARD-X evaluates models across multiple complementary dimensions:

1. **Classification Quality**:
   - **Accuracy**: Fraction of trajectories correctly classified across all 3 classes.
   - **Macro Precision / Recall / F1**: Unweighted arithmetic mean across classes (`BENIGN`, `INJECTION_RESISTED`, `HIJACKED`), preventing majority class bias.
   - **HIJACKED-Specific F1 & AUROC**: Precision-recall trade-off and area under the ROC curve for the security-critical hijacking class.
2. **Early Detection & Latency**:
   - **Prefix Evaluation ($k=1..6$)**: Models evaluated exclusively on prefix steps $1..k$ with no future-step leakage.
   - **Pre-Action Detection Rate**: Percentage of hijacked trajectories flagged with $P(\text{HIJACKED}) \ge 0.50$ prior to the execution of the final harmful action step.
   - **Detection Latency**: Number of steps between the first detection trigger and the actual deviation step ($L = \text{step}_{\text{flag}} - \text{step}_{\text{deviation}}$). Negative latency indicates advance early intervention.
3. **Uncertainty & Calibration**:
   - **Brier Score**: Multi-class quadratic penalty: $\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^3 (P_{ic} - y_{ic})^2$.
   - **Expected Calibration Error (ECE)**: 15-bin weighted absolute difference between confidence and empirical accuracy.
4. **Generalization & Robustness**:
   - **Held-Out Family Degradation**: Macro F1 gap on attack families not seen during training.
   - **Perturbation Stability**: Consistency of model predictions under additive observation noise and benign step omission.
   - **Step-Order Ablation**: Performance drop when sequence order is randomly permuted.

---

## 2. Comparative Research Matrix

All models evaluated on the official test set (Seed 42, 90 trajectories; 30 Benign, 30 Resisted, 30 Hijacked):

| Model Architecture | Accuracy | Macro F1 | HIJACKED F1 | Pre-Action Rate | Mean Latency | ECE (Calibrated) | Brier (Calibrated) | Params |
|---|---|---|---|---|---|---|---|---|
| **Mean-Pooled LogReg** *(Frozen)* | 0.6556 | 0.6456 | 0.7027 | 66.67% | -1.12 steps | 0.1820 | 0.3840 | 1,155 |
| **CLS Transformer** *(Frozen)* | 0.7111 | 0.7033 | 0.7826 | 76.67% | -1.84 steps | 0.1410 | 0.3120 | 422,403 |
| **TraceGuard LSTM** *(Reproduced Baseline)* | 0.8556 | 0.8533 | 0.9032 | 93.33% | -2.71 steps | 0.0690 | 0.2003 | 395,651 |
| **TraceGuard GRU** *(Model B1)* | **0.9333** | **0.9339** | **0.9492** | 93.33% | -2.61 steps | **0.0487** | **0.1257** | **296,835** |
| **Temporal Attention** *(Model B2)* | 0.8889 | 0.8885 | 0.9333 | 93.33% | **-3.04 steps** | 0.0518 | 0.1713 | 305,155 |
| **Multi-Task Early Warning** *(Model B3)* | 0.9222 | 0.9225 | 0.9032 | 86.67% | -2.65 steps | 0.0412 | 0.1182 | 296,964 |

---

## 3. Hypothesis Verification Summary

| Hypothesis ID | Proposed Claim | Test Method | Measured Outcome | Decision |
|---|---|---|---|---|
| **H1 (GRU Efficiency)** | GRU matches/exceeds LSTM with fewer parameters | 5-seed evaluation & parameter count | 5-seed F1 $0.8954 \pm 0.0318$ vs LSTM $0.8553 \pm 0.0354$; 25% fewer params | **SUPPORTED** |
| **H2 (Attention Latency)** | Temporal attention enables earlier detection | Paired $t$-test & Wilcoxon signed-rank | Latency -3.04 vs -2.71 steps ($p = 0.0014$, Wilcoxon $p = 0.0027$) | **SUPPORTED** |
| **H3 (Multi-Task Early Warning)** | Auxiliary head provides valid pre-deviation signals | Joint training on derived metadata | Main F1 0.9225, Auxiliary Early Warning AUROC 0.7805 | **SUPPORTED** |
| **H4 (Calibration via Scaling)** | Post-hoc temperature scaling improves ECE | Validation-fit Platt scaling | ECE reduced by 30.6% (LSTM) and 37.1% (Attention); zero rank change | **SUPPORTED** |
| **H5 (Template Generalization)** | Models overfit synthetic attack templates | Held-out attack-family evaluation | Macro F1 dropped from 0.9165 to 0.6735 on held-out `tool_misuse` | **SUPPORTED** |
| **H6 (Uncertainty Safety Gate)** | 3-way gate reduces missed harm via abstain band | Offline policy simulation on test set | Missed harm dropped from 6.67% to 5.00%; benign escalation 27.78% | **SUPPORTED** |

---

## 4. In-Depth Failure Mode & Error Analysis

### 4.1 Type-A vs Type-B Hijacking Trajectories
- **Type-A Trajectories (No Precursor, 60 dataset samples)**: Injection and deviation occur at the identical step. All models exhibit a 0.0% detection rate at step $k < d_{\text{step}}$. Detection occurs strictly at the deviation step ($L=0$).
- **Type-B Trajectories (Precursor Present, 140 dataset samples)**: An adversarial probe, prompt injection attempt, or suspicious reconnaissance step precedes the harmful action. Here, Temporal Attention excels, flagging the trajectory up to 3 steps prior to harmful tool execution.

### 4.2 Ambiguity at Early Prefix Steps ($k=1, 2$)
At steps 1 and 2, trajectories consist solely of benign user goal clarifications or routine read queries without observations. In this regime:
- Normalized entropy $H(P)$ is high ($> 0.70$).
- The Uncertainty Gate appropriately flags these as `ABSTAIN` if forced to gate at step 1, generating the measured 27.78% benign escalation rate.

### 4.3 BENIGN vs INJECTION_RESISTED Confusion
Across all models, the primary source of classification error is distinguishing between `BENIGN` and `INJECTION_RESISTED` (accounting for ~85% of off-diagonal confusion matrix errors). In both classes, the agent successfully resists hijacking; however, lexical markers of prompt injections present in observations create elevated risk scores.

---

## 5. Security & Safety Decision Boundary Summary

1. **Deterministic Fail-Closed Policy**: Any runtime detector error (OOM, missing step index, malformed JSON) immediately triggers a `BLOCK` decision, completely preventing tool execution.
2. **Tool Execution Sandboxing**: Only `ALLOW` permits tool dispatch. Under `ABSTAIN` or `BLOCK`, the tool execution method is never reached.
3. **No Dynamic Execution**: Tools operate strictly through predefined Python function dispatch (`tools.py`) with input validation. Arbitrary shell access and unconstrained network endpoints are prohibited.
