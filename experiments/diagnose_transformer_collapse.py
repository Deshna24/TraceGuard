"""
diagnose_transformer_collapse.py
Pre-training diagnostics for TRACEGUARD Transformer v4.1

Investigates why INJECTION_RESISTED -> BENIGN collapse occurred.
Checks: class balance in split, embedding separability, step-level
signal, pooling behaviour.

Run from repo root:
    python experiments/diagnose_transformer_collapse.py
"""
import json, sys, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

RESULTS = ROOT / 'results' / 'transformer_v4_1'
(RESULTS / 'reports').mkdir(parents=True, exist_ok=True)

# ── Locate data + split ──────────────────────────────────────────────────────
DATA_CANDIDATES = [
    ROOT / 'data' / 'raw' / 'trajectories' / 'traceguard_v4_1.jsonl',
    ROOT / 'traceguard' / 'data' / 'traceguard_v4_1.jsonl',
]
SPLIT_CANDIDATES = [
    ROOT / 'traceguard' / 'outputs' / 'splits' / 'split_seed42.json',
    ROOT / 'traceguard' / 'outputs' / 'handoff' / 'splits' / 'split_seed42.json',
]

data_path = next((c for c in DATA_CANDIDATES if c.exists()), None)
split_path = next((c for c in SPLIT_CANDIDATES if c.exists()), None)
assert data_path,  "Dataset not found"
assert split_path, "Split not found"

trajs = []
with open(data_path, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            trajs.append(json.loads(line))

with open(split_path) as f:
    split = json.load(f)

train_gids = set(str(x) for x in split['train_group_ids'])
val_gids   = set(str(x) for x in split['val_group_ids'])
test_gids  = set(str(x) for x in split['test_group_ids'])

traj_by_id   = {t['trajectory_id']: t for t in trajs}
group_map    = {}
for t in trajs:
    group_map.setdefault(str(t['counterfactual_group_id']), []).append(t)

def gids_to_trajs(gids):
    out = []
    for gid in gids:
        out.extend(group_map.get(gid, []))
    return out

train_trajs = gids_to_trajs(train_gids)
val_trajs   = gids_to_trajs(val_gids)
test_trajs  = gids_to_trajs(test_gids)

from collections import Counter
lines = []
SEP = '=' * 70

def section(s):
    lines.append('')
    lines.append(SEP)
    lines.append(f'## {s}')
    lines.append(SEP)
    print(f'\n{SEP}\n  {s}\n{SEP}')

def log(s):
    lines.append(s)
    print(' ', s)

# ── A. Class counts per split ─────────────────────────────────────────────────
section('A. Class Counts per Split')
for split_name, tlist in [('TRAIN', train_trajs), ('VAL', val_trajs), ('TEST', test_trajs)]:
    cnt = Counter(t['label'] for t in tlist)
    log(f'{split_name}: BENIGN={cnt["BENIGN"]}  INJECTION_RESISTED={cnt["INJECTION_RESISTED"]}  HIJACKED={cnt["HIJACKED"]}  TOTAL={len(tlist)}')

# ── B. Sequence lengths ───────────────────────────────────────────────────────
section('B. Sequence Length Distribution by Class (TRAIN)')
for cls in ['BENIGN', 'INJECTION_RESISTED', 'HIJACKED']:
    lens = [len(t['steps']) for t in train_trajs if t['label'] == cls]
    log(f'{cls}: min={min(lens)}  max={max(lens)}  mean={np.mean(lens):.2f}  std={np.std(lens):.2f}')

# ── C. Embedding generation / Loading ──────────────────────────────────────────
section('C. Loading / Generating Embeddings')
try:
    cache_candidates = [
        ROOT / 'traceguard' / 'outputs' / 'embeddings_cache' / 'embeddings_all-MiniLM-L6-v2.npy',
        ROOT / 'results' / 'embeddings' / 'embeddings_all-MiniLM-L6-v2.npy',
        ROOT / 'data' / 'embeddings' / 'v3_embeddings_all-MiniLM-L6-v2.npy',
    ]
    cache_path = next((c for c in cache_candidates if c.exists()), None)
    
    sample_trajs = []
    for cls in ['BENIGN', 'INJECTION_RESISTED', 'HIJACKED']:
        cls_trajs = [t for t in train_trajs if t['label'] == cls]
        sample_trajs.extend(cls_trajs)

    if cache_path:
        log(f'Loading cached embeddings from: {cache_path}')
        emb_cache = np.load(cache_path, allow_pickle=True).item()
    else:
        from sentence_transformers import SentenceTransformer
        EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
        log(f'Loading embedding model: {EMB_MODEL}')
        model_st = SentenceTransformer(EMB_MODEL)

        def embed_traj(t):
            texts = []
            for step in t['steps']:
                if isinstance(step, dict):
                    parts = []
                    for key in ['action', 'tool', 'observation', 'text', 'content']:
                        if key in step and step[key]:
                            parts.append(str(step[key])[:200])
                    texts.append(' | '.join(parts) if parts else str(step)[:200])
                else:
                    texts.append(str(step)[:200])
            embs = model_st.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embs  # shape: (T, 384)

        log('Embedding train trajectories...')
        emb_cache = {}
        for i, t in enumerate(sample_trajs):
            emb_cache[t['trajectory_id']] = embed_traj(t)
            if (i+1) % 50 == 0:
                log(f'  Embedded {i+1}/{len(sample_trajs)}...')

    # ── D. Mean-pooled representation similarity ─────────────────────────────
    section('D. Mean-Pooled Representation Separability (BENIGN vs INJECTION_RESISTED)')
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    mean_reps = {}
    for cls in ['BENIGN', 'INJECTION_RESISTED', 'HIJACKED']:
        cls_trajs = [t for t in sample_trajs if t['label'] == cls]
        reps = np.array([emb_cache[t['trajectory_id']].mean(axis=0) for t in cls_trajs])
        mean_reps[cls] = reps
        log(f'{cls}: mean-pool shape={reps.shape}  L2-norm mean={np.linalg.norm(reps, axis=1).mean():.4f}')

    # Cosine similarity between class centroids
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    centroid_B  = mean_reps['BENIGN'].mean(axis=0)
    centroid_IR = mean_reps['INJECTION_RESISTED'].mean(axis=0)
    centroid_H  = mean_reps['HIJACKED'].mean(axis=0)
    log(f'Centroid cosine sim BENIGN-INJECTION_RESISTED : {cosine(centroid_B, centroid_IR):.4f}')
    log(f'Centroid cosine sim BENIGN-HIJACKED           : {cosine(centroid_B, centroid_H):.4f}')
    log(f'Centroid cosine sim INJECTION_RESISTED-HIJACKED: {cosine(centroid_IR, centroid_H):.4f}')

    # L2 distance between centroids
    log(f'Centroid L2 BENIGN-INJECTION_RESISTED : {np.linalg.norm(centroid_B - centroid_IR):.4f}')
    log(f'Centroid L2 BENIGN-HIJACKED           : {np.linalg.norm(centroid_B - centroid_H):.4f}')
    log(f'Centroid L2 INJECTION_RESISTED-HIJACKED: {np.linalg.norm(centroid_IR - centroid_H):.4f}')

    # ── LDA separability ─────────────────────────────────────────────────────
    section('E. LDA Separability (mean-pooled embeddings, train sample)')
    all_X = np.vstack([mean_reps[c] for c in ['BENIGN','INJECTION_RESISTED','HIJACKED']])
    all_y = np.array([0]*len(mean_reps['BENIGN']) + [1]*len(mean_reps['INJECTION_RESISTED']) + [2]*len(mean_reps['HIJACKED']))
    lda = LinearDiscriminantAnalysis()
    lda.fit(all_X, all_y)
    lda_acc = lda.score(all_X, all_y)
    log(f'LDA in-sample accuracy on mean-pooled reps: {lda_acc:.4f}')
    log(f'(LDA > 0.85 suggests mean-pooled embeddings ARE separable)')
    log(f'(LDA ~ 0.33 suggests near-random — class collapse expected)')

    # ── F. Step-level analysis ────────────────────────────────────────────────
    section('F. Step-Level Analysis — Do BENIGN & INJECTION_RESISTED diverge?')
    log('Examining embedding drift across step positions (up to step 8)...')
    max_steps = 8
    step_diffs = []
    for step_idx in range(max_steps):
        b_vecs  = []
        ir_vecs = []
        for cls, vlist in [('BENIGN', mean_reps['BENIGN']), ('INJECTION_RESISTED', mean_reps['INJECTION_RESISTED'])]:
            for t in [t for t in sample_trajs if t['label'] == cls]:
                emb = emb_cache[t['trajectory_id']]
                if step_idx < len(emb):
                    if cls == 'BENIGN':
                        b_vecs.append(emb[step_idx])
                    else:
                        ir_vecs.append(emb[step_idx])
        if b_vecs and ir_vecs:
            b_mean  = np.mean(b_vecs, axis=0)
            ir_mean = np.mean(ir_vecs, axis=0)
            diff    = np.linalg.norm(b_mean - ir_mean)
            step_diffs.append(diff)
            log(f'  Step {step_idx+1}: L2(BENIGN_mean - INJECTION_RESISTED_mean) = {diff:.4f}')

    if step_diffs:
        early = np.mean(step_diffs[:3]) if len(step_diffs) >= 3 else step_diffs[0]
        late  = np.mean(step_diffs[-3:]) if len(step_diffs) >= 3 else step_diffs[-1]
        log(f'Early-step divergence (steps 1-3 avg) : {early:.4f}')
        log(f'Late-step  divergence (last 3 avg)    : {late:.4f}')
        if late > early * 1.5:
            log('=> Classes diverge in LATER steps — temporal order IS important')
        elif early > late * 1.5:
            log('=> Classes diverge in EARLY steps — early steps carry signal')
        else:
            log('=> Divergence is roughly uniform across steps')

    # ── G. Within-class vs between-class variance ────────────────────────────
    section('G. Within-class vs Between-class Variance (mean-pooled)')
    grand_mean = all_X.mean(axis=0)
    between_var = 0.0
    for cls_idx, cls in enumerate(['BENIGN','INJECTION_RESISTED','HIJACKED']):
        c_mean = mean_reps[cls].mean(axis=0)
        n_c = len(mean_reps[cls])
        between_var += n_c * np.linalg.norm(c_mean - grand_mean)**2
    within_var = sum(np.sum((mean_reps[cls] - mean_reps[cls].mean(axis=0))**2)
                    for cls in ['BENIGN','INJECTION_RESISTED','HIJACKED'])
    ratio = between_var / (within_var + 1e-9)
    log(f'Between-class variance : {between_var:.4f}')
    log(f'Within-class variance  : {within_var:.4f}')
    log(f'Between/Within ratio   : {ratio:.4f}')
    if ratio < 0.05:
        log('=> LOW ratio: classes are NOT well-separated in mean-pooled space')
        log('   CLS-token pooling or attention to final steps may help')
    else:
        log('=> Reasonable ratio: classes have some separation in mean-pooled space')

    # ── H. Last-step representation ───────────────────────────────────────────
    section('H. Last-step Representation Separability')
    last_reps = {}
    for cls in ['BENIGN','INJECTION_RESISTED','HIJACKED']:
        cls_trajs_s = [t for t in sample_trajs if t['label'] == cls]
        reps = np.array([emb_cache[t['trajectory_id']][-1] for t in cls_trajs_s])
        last_reps[cls] = reps
    cent_L_B  = last_reps['BENIGN'].mean(axis=0)
    cent_L_IR = last_reps['INJECTION_RESISTED'].mean(axis=0)
    cent_L_H  = last_reps['HIJACKED'].mean(axis=0)
    log(f'Last-step cosine sim BENIGN-INJECTION_RESISTED : {cosine(cent_L_B, cent_L_IR):.4f}')
    log(f'Last-step cosine sim BENIGN-HIJACKED           : {cosine(cent_L_B, cent_L_H):.4f}')
    all_X_last = np.vstack([last_reps[c] for c in ['BENIGN','INJECTION_RESISTED','HIJACKED']])
    lda_last = LinearDiscriminantAnalysis()
    lda_last.fit(all_X_last, all_y)
    log(f'LDA in-sample accuracy on last-step reps: {lda_last.score(all_X_last, all_y):.4f}')

    # ── I. Deviation step analysis ────────────────────────────────────────────
    section('I. Deviation Step Analysis')
    dev_steps = [t.get('deviation_step') for t in train_trajs if t['label'] in ['INJECTION_RESISTED','HIJACKED'] and t.get('deviation_step') is not None]
    if dev_steps:
        log(f'Deviation steps (IR+H train): min={min(dev_steps)}  max={max(dev_steps)}  mean={np.mean(dev_steps):.2f}')
        log('=> Trajectories diverge at deviation_step; before that, BENIGN ~= IR ~= H')
        log('   Mean-pooling DILUTES the post-deviation signal with pre-deviation noise')
        log('   CLS token sees the FULL sequence and can weight later steps adaptively')
    else:
        log('No deviation_step field found in trajectories')

    # ── J. Diagnosis summary ──────────────────────────────────────────────────
    section('J. Diagnosis Summary & Recommendations')
    log(f'LDA accuracy (mean-pooled)  : {lda_acc:.4f}')
    log(f'B/W variance ratio          : {ratio:.4f}')
    log(f'Centroid sim B-IR           : {cosine(centroid_B, centroid_IR):.4f}')
    log('')
    log('ROOT CAUSE HYPOTHESIS:')
    log('  1. BENIGN and INJECTION_RESISTED share identical EARLY steps.')
    log('     They only diverge at/after deviation_step (~step 5).')
    log('  2. Mean-pooling averages early (undifferentiated) + late (signal) steps,')
    log('     diluting the discriminative signal for INJECTION_RESISTED.')
    log('  3. HIJACKED has a strong late-sequence signal (malicious actions)')
    log('     so it remains detectable even under mean-pooling.')
    log('')
    log('FIX:')
    log('  Use CLS-token pooling: a learnable prefix token that attends')
    log('  over the full sequence and can weight the most discriminative')
    log('  steps (post-deviation) more heavily.')
    log('  Also: lower LR (3e-4) + more patience (8) allows longer convergence.')

except ImportError as e:
    log(f'sentence_transformers not available: {e}')
    log('Running structural diagnostics only (no embedding analysis).')

    section('Structural Diagnostics (no embeddings)')
    for split_name, tlist in [('TRAIN', train_trajs), ('VAL', val_trajs), ('TEST', test_trajs)]:
        cnt = Counter(t['label'] for t in tlist)
        log(f'{split_name}: B={cnt["BENIGN"]} IR={cnt["INJECTION_RESISTED"]} H={cnt["HIJACKED"]}')
    for cls in ['BENIGN', 'INJECTION_RESISTED', 'HIJACKED']:
        lens = [len(t['steps']) for t in train_trajs if t['label'] == cls]
        log(f'{cls} seq lens: min={min(lens)} max={max(lens)} mean={np.mean(lens):.2f}')
    dev_steps = [t.get('deviation_step') for t in train_trajs
                 if t['label'] in ['INJECTION_RESISTED','HIJACKED'] and t.get('deviation_step') is not None]
    if dev_steps:
        log(f'Deviation steps: min={min(dev_steps)} max={max(dev_steps)} mean={np.mean(dev_steps):.2f}')

# ── Write report ──────────────────────────────────────────────────────────────
report_path = RESULTS / 'reports' / 'pretraining_diagnostics.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('# Pre-Training Diagnostics — TRACEGUARD Transformer v4.1\n\n')
    f.write('**Purpose:** Investigate INJECTION_RESISTED → BENIGN collapse in previous run.\n\n')
    for line in lines:
        f.write(line + '\n')

print(f'\n[Diagnostics written to: {report_path}]')
