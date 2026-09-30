# Pre-Training Diagnostics — TRACEGUARD Transformer v4.1

**Purpose:** Investigate INJECTION_RESISTED → BENIGN collapse in previous run.


======================================================================
## A. Class Counts per Split
======================================================================
TRAIN: BENIGN=140  INJECTION_RESISTED=140  HIJACKED=140  TOTAL=420
VAL: BENIGN=30  INJECTION_RESISTED=30  HIJACKED=30  TOTAL=90
TEST: BENIGN=30  INJECTION_RESISTED=30  HIJACKED=30  TOTAL=90

======================================================================
## B. Sequence Length Distribution by Class (TRAIN)
======================================================================
BENIGN: min=6  max=6  mean=6.00  std=0.00
INJECTION_RESISTED: min=6  max=6  mean=6.00  std=0.00
HIJACKED: min=6  max=6  mean=6.00  std=0.00

======================================================================
## C. Loading / Generating Embeddings
======================================================================
Loading cached embeddings from: C:\Users\DESHNA\TraceGuard\traceguard\outputs\embeddings_cache\embeddings_all-MiniLM-L6-v2.npy

======================================================================
## D. Mean-Pooled Representation Separability (BENIGN vs INJECTION_RESISTED)
======================================================================
BENIGN: mean-pool shape=(140, 384)  L2-norm mean=0.9276
INJECTION_RESISTED: mean-pool shape=(140, 384)  L2-norm mean=0.9224
HIJACKED: mean-pool shape=(140, 384)  L2-norm mean=0.9102
Centroid cosine sim BENIGN-INJECTION_RESISTED : 0.9980
Centroid cosine sim BENIGN-HIJACKED           : 0.9855
Centroid cosine sim INJECTION_RESISTED-HIJACKED: 0.9919
Centroid L2 BENIGN-INJECTION_RESISTED : 0.0353
Centroid L2 BENIGN-HIJACKED           : 0.0949
Centroid L2 INJECTION_RESISTED-HIJACKED: 0.0711

======================================================================
## E. LDA Separability (mean-pooled embeddings, train sample)
======================================================================
LDA in-sample accuracy on mean-pooled reps: 1.0000
(LDA > 0.85 suggests mean-pooled embeddings ARE separable)
(LDA ~ 0.33 suggests near-random — class collapse expected)

======================================================================
## F. Step-Level Analysis — Do BENIGN & INJECTION_RESISTED diverge?
======================================================================
Examining embedding drift across step positions (up to step 8)...
  Step 1: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.0000
  Step 2: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.1808
  Step 3: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.0401
  Step 4: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.0580
  Step 5: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.0000
  Step 6: L2(BENIGN_mean - INJECTION_RESISTED_mean) = 0.0000
Early-step divergence (steps 1-3 avg) : 0.0736
Late-step  divergence (last 3 avg)    : 0.0193
=> Classes diverge in EARLY steps — early steps carry signal

======================================================================
## G. Within-class vs Between-class Variance (mean-pooled)
======================================================================
Between-class variance : 0.7145
Within-class variance  : 225.6598
Between/Within ratio   : 0.0032
=> LOW ratio: classes are NOT well-separated in mean-pooled space
   CLS-token pooling or attention to final steps may help

======================================================================
## H. Last-step Representation Separability
======================================================================
Last-step cosine sim BENIGN-INJECTION_RESISTED : 1.0000
Last-step cosine sim BENIGN-HIJACKED           : 0.9579
LDA in-sample accuracy on last-step reps: 0.6643

======================================================================
## I. Deviation Step Analysis
======================================================================
Deviation steps (IR+H train): min=5  max=5  mean=5.00
=> Trajectories diverge at deviation_step; before that, BENIGN ~= IR ~= H
   Mean-pooling DILUTES the post-deviation signal with pre-deviation noise
   CLS token sees the FULL sequence and can weight later steps adaptively

======================================================================
## J. Diagnosis Summary & Recommendations
======================================================================
LDA accuracy (mean-pooled)  : 1.0000
B/W variance ratio          : 0.0032
Centroid sim B-IR           : 0.9980

ROOT CAUSE HYPOTHESIS:
  1. BENIGN and INJECTION_RESISTED share identical EARLY steps.
     They only diverge at/after deviation_step (~step 5).
  2. Mean-pooling averages early (undifferentiated) + late (signal) steps,
     diluting the discriminative signal for INJECTION_RESISTED.
  3. HIJACKED has a strong late-sequence signal (malicious actions)
     so it remains detectable even under mean-pooling.

FIX:
  Use CLS-token pooling: a learnable prefix token that attends
  over the full sequence and can weight the most discriminative
  steps (post-deviation) more heavily.
  Also: lower LR (3e-4) + more patience (8) allows longer convergence.
