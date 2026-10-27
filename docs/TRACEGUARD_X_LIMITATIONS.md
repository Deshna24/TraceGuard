# TRACEGUARD-X Limitations, Boundary Conditions & Open Challenges

**Status:** COMPLETE (Phase 12)  
**Classification:** Scientific Limitations & Threat Modeling Document

---

## 1. Synthetic Dataset Constraints & Distributional Shift

1. **Fixed Sequence Length ($T=6$)**:
   - TRACEGUARD v4.1 comprises exactly 600 synthetic trajectories, each strictly structured to 6 sequential steps.
   - Real-world autonomous agents engage in variable-length trajectories ranging from 1 to 50+ actions. Models trained exclusively on $T=6$ sequences have not been evaluated on long-horizon looping or early single-step prompt injections.
2. **Synthetic Counterfactual Templates**:
   - Although counterfactual groups control for goal bias, the underlying text templates were synthetically generated using structured prompt frames.
   - Attack vectors are cleanly delineated into Type-A (instant deviation) and Type-B (precursor deviation), whereas real-world adversarial attacks often involve complex, multi-stage social engineering, steganographic payloads, and ambiguous compliance.

---

## 2. Attack-Family Generalization Vulnerability (Empirically Proven in Track C)

- In Track C3, when trained on 4 attack families and tested on an unseen 5th family (`tool_misuse`), the model's Macro F1 collapsed from **0.9165** to **0.6735** ($\Delta = -0.2430$).
- **Crucial Research Takeaway**: High benchmark test F1 (0.93+) on group-aware splits does **not** provide zero-day robustness against novel attack vectors. Sequential deep-learning models remain susceptible to template-level lexical shortcuts.
- **Security Boundary**: TRACEGUARD-X should be deployed as a defense-in-depth heuristic layer, **never** as a standalone infallible barrier against sophisticated adversaries.

---

## 3. Attention Interpretability: Non-Causal Nature

- Attention weights $\alpha_t$ extracted from `TraceGuardTemporalAttention` represent internal representation pooling weights across hidden states $h_t$.
- **Not Causal Explanations**: A high attention weight at step 3 indicates that the context vector $c$ draws heavily from hidden state $h_3$. It does **not** prove that step 3 caused the final classification, nor does it guarantee that altering step 3 would reverse the model's decision.
- Attention visualizations in the frontend dashboard are strictly diagnostic aids for researchers, not formal cryptographic or causal audit trails.

---

## 4. Multi-Task Supervision Feasibility & Horizon Constraints

- In Track B3, the auxiliary imminent deviation head ($y_{\text{warn}} \in [0, 1]$) relies on the existence of a documented `deviation_step` in trajectory metadata.
- For datasets lacking ground-truth deviation timestamps, supervising this head without leaking future trajectory labels is mathematically ill-posed.
- Furthermore, the horizon was fixed at $h \le 2$ steps. Adaptive or unbounded horizon prediction remains an open research problem.

---

## 5. Operational Trade-Offs of the Uncertainty-Aware Safety Gate

- The 3-way `UncertaintyActionGate` improves harmful intervention from 93.33% to 95.00% (reducing missed attacks by 25%), but introduces an **operator escalation rate of 27.78% on benign trajectories**.
- In high-throughput autonomous systems, escalating >25% of benign agent actions to human operators introduces severe operational friction and potential alert fatigue.
- Parameter selection ($\tau_{\text{block}}=0.50, \tau_{\text{abstain}}=0.25, \tau_{\text{entropy}}=0.70$) was optimized for safety on validation data, but production deployments must recalibrate these thresholds according to their specific cost ratio between false blocks and human intervention costs.

---

## 6. Pretrained Representation Freezing

- Canonical embeddings are generated via `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and frozen.
- While computationally lightweight and reproducible on CPU, MiniLM was pretrained on general natural language sentence pairs, not tool-use JSON payloads, AST structures, or agent state transitions. Fine-tuning the representation encoder end-to-end remains an unaddressed optimization avenue due to compute and sample-size constraints.

---

## 7. Sandboxed Environment & Tool Surface

- The runtime environment purposefully implements controlled, sandboxed tools:
  - `search`: In-memory dictionary retrieval.
  - `calculator`: Restricted AST arithmetic evaluation.
  - `database`: Mock in-memory records.
- The system intentionally does not provide arbitrary Bash execution, unrestricted SQL execution, or live external network endpoints. Consequently, tool-level exploit payloads (e.g., shell command injection, SQL injection) were not part of the threat model; the focus is exclusively on high-level behavioral hijacking of agent goals.
