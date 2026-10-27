# TRACEGUARD-X System & Research Architecture

**Status:** COMPLETE (Phase 12)  
**Classification:** New Experimental Extension Architecture & System Specification

---

## 1. Architectural Philosophy & Isolation Guarantees

TRACEGUARD-X is an isolated, uncertainty-aware deep learning research extension built directly atop the verified TRACEGUARD baseline. It was designed under four strict architectural guarantees:

1. **Non-Negotiable Baseline Preservation**: Frozen artifacts (dataset `traceguard_v4_1.jsonl`, checkpoint `lstm_seed42_best.pth`, partition `split_seed42.json`, and official benchmarks) are protected and continuously hash-verified via SHA-256 (`traceguard_x.integrity.verify_integrity`, 166/166 files intact).
2. **Complete Namespace Isolation**: All new architectures, experiment runners, calibration transformers, and safety gate policies reside strictly within `traceguard_x/`, completely isolated from `traceguard/`.
3. **Execution Boundary Integrity**: Neither the original nor the experimental gate allows tool execution if an action is blocked or abstained. Sandboxed tools (`search`, `calculator`, `database`) execute only under an explicit `ALLOW`. Any model or detector exception triggers an immediate fail-closed `BLOCK`.
4. **Strict Causal Prefix Invariance**: Temporal inference at prefix step $k$ ($1 \le k \le 6$) has access exclusively to steps $1 \dots k$. Future steps ($>k$) are strictly inaccessible.

---

## 2. Deep Learning Model Architectures

All models operate on the canonical 384-dimensional sentence embeddings generated step-by-step by `sentence-transformers/all-MiniLM-L6-v2`. Input tensors are shaped `[batch, max_seq_len, 384]`.

```
                    Trajectory Prefix Steps (1..k)
                                  │
                                  ▼
                     [MiniLM-L6-v2 (Frozen 384d)]
                                  │
                                  ▼
                   Padded Sequence [B, T, 384]
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   TraceGuardGRU          Temporal Attention        Multi-Task Early Warning
   (Model B1)                (Model B2)                    (Model B3)
  2-layer GRU (128)         2-layer GRU (128)             2-layer GRU (128)
Last Hidden State         Step Attention Pooling       Last Hidden State
  Linear(128 → 3)          Linear(128 → 3)        ┌───────┴───────┐
        │                         │               ▼               ▼
        │                         │           Main Head     Auxiliary Head
        │                         │         Linear(128→3)   Linear(128→1)
        ▼                         ▼               │               │
  Logits [B, 3]             Logits [B, 3]         ▼               ▼
  (Params: 296,835)       (Params: 305,155)  Logits [B, 3]   Warning Logit [B, 1]
                                            (Params: 296,964)
```

### 2.1 Frozen Baseline: TraceGuardLSTM
- **Implementation**: `traceguard.src.model_lstm.TraceGuardLSTM`
- **Parameters**: 395,651
- **Architecture**: Unidirectional 2-layer LSTM, input size 384, hidden size 128, dropout 0.2, packed sequence representation, pooling via final hidden state $h_T^{(2)}$, output layer `Linear(128 → 3)`.
- **Role**: Reference baseline for all comparative benchmarks.

### 2.2 Model B1: TraceGuardGRU
- **Implementation**: `traceguard_x.models.gru.TraceGuardGRU`
- **Parameters**: 296,835 (~25.0% fewer parameters than baseline LSTM)
- **Architecture**: Unidirectional 2-layer GRU (`batch_first=True`, hidden size 128, dropout 0.2).
- **Mechanism**: Sequences are packed using `pack_padded_sequence` and unpacked via `pad_packed_sequence`. The final valid hidden step for each sequence is extracted via index gathering `h[batch_idx, lengths - 1]`, passed through a 0.2 dropout layer, and projected via `Linear(128 → 3)`.
- **Design Rationale**: Eliminates the separate cell state $c_t$ of LSTMs, reducing memory footprint and parameter count while preserving sequential modeling fidelity.

### 2.3 Model B2: TraceGuardTemporalAttention
- **Implementation**: `traceguard_x.models.attention.TraceGuardTemporalAttention`
- **Parameters**: 305,155
- **Architecture**:
  - **Encoder**: 2-layer unidirectional GRU ($h_t \in \mathbb{R}^{128}$).
  - **Attention Network**: Additive step-level scoring network:
    $$u_t = \tanh(W_a h_t + b_a), \quad u_t \in \mathbb{R}^{64}$$
    $$\alpha_t = \frac{\exp(v_a^\top u_t)}{\sum_{\tau=1}^k \exp(v_a^\top u_\tau)}$$
  - **Dynamic Padding Mask**: Large negative value ($-10^9$) is applied to padded steps ($t > k$) before softmax normalization.
  - **Representation**: Context vector $c = \sum_{t=1}^k \alpha_t h_t$, passed through `Dropout(0.2)` and `Linear(128 → 3)`.
- **Diagnostic Notice**: Attention weights $\alpha_t$ represent representation pooling weights and are visualized as diagnostic indicators of temporal salience, **not** causal explanations.

### 2.4 Model B3: TraceGuardMultiTaskEarlyWarning
- **Implementation**: `traceguard_x.models.multitask.TraceGuardMultiTaskEarlyWarning`
- **Parameters**: 296,964
- **Architecture**:
  - Shared 2-layer GRU backbone ($h_t \in \mathbb{R}^{128}$).
  - **Main Classification Head**: `Linear(128 → 3)` predicting 3-class trajectory status.
  - **Auxiliary Early-Warning Head**: `Linear(128 → 1)` predicting whether a behavioral hijacking deviation is imminent within horizon $h \le 2$ steps:
    $$y_{\text{warn}}^{(k)} = \mathbb{I}(\text{deviation occurs at step } d \text{ with } k \le d \le k + h)$$
- **Supervision Derivation**: Derived strictly from dataset metadata (`deviation_step`) during training without leaking future step text or future tool outputs into the prefix input tensor.
- **Joint Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{warn}}, y_{\text{warn}})$, with $\lambda = 0.5$.

### 2.5 Architectural Scoping Note on Model B4 (Temporal CNN)
A dilated 1D Temporal Convolutional Network (TCN) was scoped and analyzed during Phase 2. Because TRACEGUARD trajectories are strictly fixed at $T=6$ steps, the receptive field required is extremely shallow ($2^1=2, 2^2=4$). Recurrent and attention pooling models already achieve saturation on $T \le 6$, while dilated convolutions would introduce padding edge-artifacts on ultra-short prefixes ($k=1, 2$). Consistent with Section 6 (Model B4 guidelines), unnecessary architectural complexity was avoided in favor of rigorous statistical evaluation of Models B1, B2, and B3.

---

## 3. Input Representation & Prefix Caching

### 3.1 Canonical Trajectory Text Representation
Steps are formatted via `traceguard.src.preprocessing.build_step_text` according to the canonical template:
```text
ORIGINAL USER GOAL: {user_goal}
CURRENT STEP: {step_number}
ACTION: {action}
TOOL: {tool}
TOOL INPUT: {tool_input}
TOOL OBSERVATION: {tool_observation}
STATE: {state}
```
Forbidden metadata fields (`label`, `counterfactual_group_id`, `deviation_step`, `precursor_step`, `attack_family`, `attack_subtype`) are strictly excluded and asserted.

### 3.2 Dynamic Runtime Prefix Caching
During live agent execution:
1. When candidate action step $k$ is evaluated before execution, it contains a null observation.
2. Once the tool executes, the observation is populated, and step $k$ becomes an authoritative completed step.
3. Precomputed step embeddings are cached under the tuple key `(trajectory_id_or_goal, step_index, is_completed)` to eliminate redundant sentence transformer inferences while preventing cross-trajectory contamination.

---

## 4. Safety Gate Architectures

```
                      Proposed Tool Action Step k
                                   │
                                   ▼
                       TraceGuard-X Runtime Detector
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         Predicted Probabilities           Entropy H(P)
             P(HIJACKED)             (Normalized to [0, 1])
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                     Configured Action Gate
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
     PreActionGate                               UncertaintyActionGate
   (Standard Baseline)                             (TraceGuard-X 3-Way)
          │                                                 │
  P(HIJACKED) >= 0.50?                              P(HIJACKED) >= 0.50?
     ├── YES ──► BLOCK                                 ├── YES ──► BLOCK
     └── NO  ──► ALLOW                                 └── NO   │
                                                    P(HIJACKED) >= 0.25 OR H(P) >= 0.70?
                                                       ├── YES ──► ABSTAIN (Escalate)
                                                       └── NO  ──► ALLOW
                                                            │
                                              Fail-Closed Safety Guard:
                                              If Detector Fails ──► BLOCK
                                              If Unknown Tool   ──► BLOCK
```

### 4.1 Baseline PreActionGate
- **Implementation**: `traceguard.runtime.action_gate.PreActionGate`
- **Policy**:
  - `BLOCK` if $P(\text{HIJACKED}) \ge 0.50$.
  - `ALLOW` otherwise.
- **Fail-Closed**: Any detector exception results in `BLOCK`.

### 4.2 Experimental UncertaintyActionGate
- **Implementation**: `traceguard_x.gate.uncertainty_gate.UncertaintyActionGate`
- **Policy Formulation**:
  1. Compute normalized Shannon entropy over 3 classes:
     $$H(P) = -\frac{1}{\ln 3} \sum_{c=1}^3 P(c) \ln (P(c) + \epsilon)$$
  2. Decision logic:
     $$\text{Decision} = \begin{cases} 
     \text{BLOCK}, & \text{if } P(\text{HIJACKED}) \ge \tau_{\text{block}} \; (0.50) \\
     \text{ABSTAIN}, & \text{if } P(\text{HIJACKED}) \ge \tau_{\text{abstain}} \; (0.25) \text{ or } H(P) \ge \tau_{\text{entropy}} \; (0.70) \\
     \text{ALLOW}, & \text{otherwise}
     \end{cases}$$
- **Execution Boundary Guarantee**:
  - Under `ALLOW`: The sandboxed tool executes normally.
  - Under `BLOCK`: Raises `BlockedActionError`; tool execution is halted.
  - Under `ABSTAIN`: Tool execution is halted; action is escalated to operator review without tool side-effects.
  - Under any detector failure or unregistered tool name: Fails closed (`BLOCK`).

---

## 5. Web & Runtime Integration Architecture

### 5.1 Backend (`traceguard/web/server.py`)
- **FastAPI Application**:
  - `POST /api/run/start`: Accepts `RunStartRequest` containing `model_architecture` (`"lstm"`, `"gru"`, `"attention"`, `"multitask"`) and `gate_type` (`"standard"`, `"uncertainty"`).
  - Defaults strictly to `"lstm"` and `"standard"` to preserve legacy compatibility.
  - `GET /api/experiments/summary`: Serves machine-readable benchmark comparisons directly from verified experimental result artifacts.
- **Agent Runner (`traceguard/web/agent_runner.py`)**:
  - Dynamically binds the chosen detector and gate during run initialization.
  - Streams typed WebSocket events (`step_started`, `model_evaluated`, `action_gated`, `step_completed`, `run_finished`).
  - Emits auxiliary experimental fields (`warning_score`, `attention_weights`, `entropy`, `abstain_reason`) seamlessly.

### 5.2 Frontend Dashboard (`frontend/src/`)
- Built with React 19 and Vite.
- **Dashboard (`frontend/src/pages/Dashboard.tsx`)**:
  - Interactive selector pills for Model Architecture (Baseline LSTM, GRU, Temporal Attention, Multi-Task Early Warning) and Safety Gate (Standard Threshold, Uncertainty-Aware 3-Way).
  - Live timeline showing step risk, proposed tool, attention distribution, and gate intervention state.
- **Evidence Page (`frontend/src/pages/Evidence.tsx`)**:
  - Full research comparison table with verified metrics.
  - Calibration summary panel with temperature scaling parameters and ECE reductions.
  - Safety gate operational trade-off metrics.
  - Research limitations and dataset boundary banner.
