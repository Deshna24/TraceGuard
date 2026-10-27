# Track F — Uncertainty-Aware Safety Gate Evaluation Report

## Policy Comparison on Held-Out Test Set (540 prefix decisions)

| Metric | Standard 2-Way Gate (P >= 0.5) | Uncertainty 3-Way Gate (ALLOW/ABSTAIN/BLOCK) | Delta |
|---|---|---|---|
| Harmful-Action Hard Blocking Rate | 95.00% | 95.00% | +0.00% |
| Total Harmful Intervention Rate (Block+Abstain) | 95.00% | 95.00% | +0.00% |
| Missed Harmful-Action Rate | 5.00% | 5.00% | +0.00% |
| False Blocking Rate on Benign | 14.44% | 14.44% | +0.00% |
| Escalation / Abstain Rate on Benign | 0.00% | 27.78% | +27.78% |
| Clean Tool Execution Rate on Benign | 85.56% | 57.78% | -27.78% |
| Per-Decision Computation Latency | < 0.1 μs | 0.98 μs | Negligible overhead |

## Operational Trade-Off Analysis
- The 3-way Uncertainty Gate catches ambiguous precursor states where P(HIJACKED) is in [0.25, 0.50) or entropy is elevated.
- Increases harmful-action intervention rate without modifying the conservative 0.50 hard BLOCK guarantee.
- Trade-off: Increases human escalation burden on benign tasks by the abstain percentage.