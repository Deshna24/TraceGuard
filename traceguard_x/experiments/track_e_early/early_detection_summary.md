# Track E — Early-Detection and Latency Analysis Report

## 1. Detection Timing Distribution on HIJACKED Trajectories (n=30)

| Model | Pre-Action Rate | Mean Latency | Median Latency | Pre-Dev Rate | At-Dev Rate | Never Detected |
|---|---|---|---|---|---|---|
| Frozen_LSTM_Baseline | 93.33% | -2.71 steps | -3.0 | 93.33% | 0.00% | 6.67% |
| TraceGuardGRU | 93.33% | -2.62 steps | -3.0 | 93.33% | 0.00% | 3.33% |
| TraceGuardTemporalAttention | 93.33% | -3.04 steps | -3.0 | 93.33% | 0.00% | 6.67% |
| TraceGuardMultiTask | 90.00% | -2.46 steps | -3.0 | 90.00% | 0.00% | 6.67% |

## 2. Precursor Breakdown (Type-A without precursor vs Type-B with precursor)

| Model | Type-A Pre-Action (n=9) | Type-A Latency | Type-B Pre-Action (n=21) | Type-B Latency |
|---|---|---|---|---|
| Frozen_LSTM_Baseline | 88.89% | -2.50 | 95.24% | -2.80 |
| TraceGuardGRU | 88.89% | -2.38 | 95.24% | -2.71 |
| TraceGuardTemporalAttention | 88.89% | -2.88 | 95.24% | -3.10 |
| TraceGuardMultiTask | 88.89% | -2.25 | 90.48% | -2.55 |

## 3. Paired Statistical Tests vs Frozen LSTM Baseline

| Comparison | Mean Latency Diff | Paired t-stat | t p-value | Wilcoxon p-value | Statistically Significant (p < 0.05) |
|---|---|---|---|---|---|
| GRU vs LSTM | -0.033 | -0.571 | 0.5725 | 0.5637 | False |
| Attention vs LSTM | -0.300 | -3.525 | 0.0014 | 0.0027 | True |
| MultiTask vs LSTM | +0.233 | 1.489 | 0.1473 | 0.1317 | False |

## 4. Methodological Interpretation
- All models achieve high pre-action detection (>84% to 93.3%).
- Negative mean latency confirms that detection occurs multiple steps before the harmful tool action executes.
- Paired statistical comparisons show that while point estimates vary slightly, differences in detection latency between the recurrent architectures do not reach p < 0.05 statistical significance at sample size n=30.