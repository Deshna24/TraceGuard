"""
run_transformer_v4_1_cls.py  — TRACEGUARD Transformer v4.1 (CLS-token pooling)

CHANGES vs previous run:
  - Learnable CLS token prepended to every sequence
  - CLS representation used for classification (not mean-pool)
  - LR lowered: 1e-3 -> 3e-4
  - Patience increased: 7 -> 8
  - Gradient clipping unchanged (max_norm=1.0)
  - Architecture otherwise identical to spec

Usage:
    python experiments/run_transformer_v4_1_cls.py --seed 42
"""
import os, json, argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

RESULTS = ROOT / 'results' / 'transformer_v4_1'
SUBDIRS = ['checkpoints','metrics','plots','confusion_matrices','prefix_detection',
           'latency','ablations','robustness','splits','embeddings','configs','reports','handoff']
for d in SUBDIRS:
    (RESULTS / d).mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
EMB_MODEL_NAME  = 'sentence-transformers/all-MiniLM-L6-v2'
EMB_DIM         = 384
MODEL_DIM       = 128
N_LAYERS        = 2
N_HEADS         = 4
FF_DIM          = 256
DROPOUT         = 0.2
NUM_CLASSES     = 3
LABEL_MAP       = {'BENIGN': 0, 'INJECTION_RESISTED': 1, 'HIJACKED': 2}
INV_LABEL_MAP   = {0: 'BENIGN', 1: 'INJECTION_RESISTED', 2: 'HIJACKED'}
CLASS_NAMES     = ['BENIGN', 'INJECTION_RESISTED', 'HIJACKED']
THRESHOLD       = 0.5
BATCH_SIZE      = 32
MAX_EPOCHS      = 50
PATIENCE        = 8
LR              = 3e-4
WEIGHT_DECAY    = 1e-4

DATA_CANDIDATES = [
    ROOT / 'data' / 'raw' / 'trajectories' / 'traceguard_v4_1.jsonl',
    ROOT / 'traceguard' / 'data' / 'traceguard_v4_1.jsonl',
]
SPLIT_CANDIDATES = [
    ROOT / 'traceguard' / 'outputs' / 'splits' / 'split_seed42.json',
    ROOT / 'traceguard' / 'outputs' / 'handoff' / 'splits' / 'split_seed42.json',
]


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


# ── Transformer with learnable CLS token ─────────────────────────────────────
class CLSTransformer(nn.Module):
    """
    Architecture:
        input_dim=384  →  Linear  →  model_dim=128
        Prepend learnable CLS token (model_dim)
        Apply sinusoidal PositionalEncoding over [CLS, step1, ..., stepN]
        TransformerEncoder (2 layers, 4 heads, ff=256, dropout=0.2)
        Extract CLS position (index 0)
        Dropout  →  Linear(model_dim, 3)

    The CLS token can attend to all step positions via multi-head self-attention,
    allowing it to weight the most discriminative (post-deviation) steps.
    """
    def __init__(self, input_dim=384, model_dim=128, num_heads=4,
                 ff_dim=256, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.model_dim   = model_dim
        self.input_proj  = nn.Linear(input_dim, model_dim)
        # Learnable CLS token (1 × model_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_dropout  = nn.Dropout(dropout)
        self.classifier   = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: list) -> torch.Tensor:
        """
        x       : (B, T, input_dim)  padded step embeddings
        lengths : list[int]          actual sequence lengths (before padding)

        Returns logits: (B, num_classes)
        """
        B, T, _ = x.shape
        # Project to model_dim
        x = self.input_proj(x)                            # (B, T, model_dim)

        # Expand CLS and prepend
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, model_dim)
        x = torch.cat([cls_tokens, x], dim=1)             # (B, T+1, model_dim)

        # Positional encoding (includes CLS at position 0)
        x = self.pos_encoder(x)                           # (B, T+1, model_dim)

        # Padding mask: True = pad position (ignored by attention)
        # CLS is at position 0 and is never masked
        # Positions 1..T are real; positions 1..length[i] are real, rest are pad
        device = x.device
        # shape: (B, T+1)
        mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            # positions L+1 .. T are padding (after the CLS offset)
            if L < T:
                mask[i, L + 1:] = True

        # Encode
        out = self.transformer_encoder(x, src_key_padding_mask=mask)  # (B, T+1, model_dim)

        # CLS representation is at position 0
        cls_out = out[:, 0, :]                             # (B, model_dim)
        cls_out = self.cls_dropout(cls_out)
        return self.classifier(cls_out)                    # (B, num_classes)


# ── Data utilities ────────────────────────────────────────────────────────────
def load_dataset_local(data_path):
    trajs = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                trajs.append(json.loads(line))
    assert len(trajs) == 600, f"Expected 600 trajectories, got {len(trajs)}"
    return trajs

def load_split(split_path):
    with open(split_path) as f:
        return json.load(f)

def embed_trajectory(t, model_st):
    texts = []
    for step in t['steps']:
        if isinstance(step, dict):
            parts = []
            for key in ['action', 'tool', 'observation', 'text', 'content']:
                if key in step and step[key]:
                    parts.append(str(step[key])[:300])
            texts.append(' | '.join(parts) if parts else str(step)[:300])
        else:
            texts.append(str(step)[:300])
    return model_st.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def build_emb_dict(trajs, seed):
    """Generate or load trajectory embeddings."""
    cache_candidates = [
        RESULTS / 'embeddings' / f'embeddings_{EMB_MODEL_NAME.replace("/","_")}.npy',
        ROOT / 'traceguard' / 'outputs' / 'embeddings_cache' / 'embeddings_all-MiniLM-L6-v2.npy',
        ROOT / 'data' / 'embeddings' / 'v3_embeddings_all-MiniLM-L6-v2.npy',
    ]
    cache_path = next((c for c in cache_candidates if c.exists()), None)
    if cache_path:
        print(f'[Embeddings] Loading cache: {cache_path}')
        emb_dict = np.load(cache_path, allow_pickle=True).item()
        # Ensure saved to RESULTS/embeddings for artifact completeness
        dest = RESULTS / 'embeddings' / f'embeddings_{EMB_MODEL_NAME.replace("/","_")}.npy'
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, emb_dict)
            print(f'[Embeddings] Copied cache to {dest}')
    else:
        print(f'[Embeddings] Generating with {EMB_MODEL_NAME} ...')
        from sentence_transformers import SentenceTransformer
        model_st = SentenceTransformer(EMB_MODEL_NAME)
        emb_dict = {}
        for i, t in enumerate(trajs):
            emb_dict[t['trajectory_id']] = embed_trajectory(t, model_st)
            if (i + 1) % 100 == 0:
                print(f'  Embedded {i+1}/{len(trajs)}')
        dest = RESULTS / 'embeddings' / f'embeddings_{EMB_MODEL_NAME.replace("/","_")}.npy'
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, emb_dict)
        print(f'[Embeddings] Saved to {dest}')
    return emb_dict


class TrajectoryDataset(Dataset):
    def __init__(self, trajs, emb_dict):
        self.trajs    = trajs
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        t      = self.trajs[idx]
        tid    = t['trajectory_id']
        emb    = self.emb_dict[tid]                             # (T, 384)
        label  = LABEL_MAP[t['label']]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long), tid


def collate_fn(batch):
    embs, labels, tids = zip(*batch)
    lengths  = [e.shape[0] for e in embs]
    max_len  = max(lengths)
    dim      = embs[0].shape[1]
    padded   = torch.zeros(len(batch), max_len, dim)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0], :] = e
    return padded, torch.stack(labels), lengths, list(tids)


def make_loader(trajs, emb_dict, shuffle, seed):
    ds = TrajectoryDataset(trajs, emb_dict)
    g  = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate_fn, generator=g if shuffle else None)


# ── Evaluation helpers ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    acc             = float(accuracy_score(y_true, y_pred))
    prec_p, rec_p, f1_p, sup_p = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
    prec_m, rec_m, f1_m, _     = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)

    per_class = {}
    for i, cls in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        try:
            auroc = float(roc_auc_score(y_bin, y_prob[:, i]))
        except Exception:
            auroc = None
        try:
            pr_auc = float(average_precision_score(y_bin, y_prob[:, i]))
        except Exception:
            pr_auc = None
        per_class[cls] = {
            'precision': float(prec_p[i]), 'recall': float(rec_p[i]),
            'f1': float(f1_p[i]), 'support': int(sup_p[i]),
            'auroc': auroc, 'pr_auc': pr_auc,
        }

    return {
        'accuracy': acc,
        'macro_precision': float(prec_m), 'macro_recall': float(rec_m), 'macro_f1': float(f1_m),
        'per_class': per_class,
    }


def plot_cm(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Transformer (CLS) Confusion Matrix — {tag}')
    plt.tight_layout()
    out = RESULTS / 'confusion_matrices' / f'{tag}_confusion_matrix.png'
    plt.savefig(out, dpi=150); plt.close()
    return out


def save_predictions(y_true, y_pred, y_prob, tids, tag):
    path = RESULTS / 'metrics' / f'{tag}_predictions.csv'
    rows = []
    for tid, yt, yp, ypr in zip(tids, y_true, y_pred, y_prob):
        rows.append({'trajectory_id': tid, 'true_label': CLASS_NAMES[int(yt)],
                     'pred_label': CLASS_NAMES[int(yp)],
                     'p_benign': round(float(ypr[0]),6),
                     'p_resisted': round(float(ypr[1]),6),
                     'p_hijacked': round(float(ypr[2]),6)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── Main run ──────────────────────────────────────────────────────────────────
def run(seed=42):
    # Seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Locate data & split
    data_path  = next((c for c in DATA_CANDIDATES if c.exists()), None)
    split_path = next((c for c in SPLIT_CANDIDATES if c.exists()), None)
    assert data_path,  "traceguard_v4_1.jsonl not found"
    assert split_path, "split_seed42.json not found"

    trajs = load_dataset_local(data_path)
    split = load_split(split_path)

    train_gids = set(str(x) for x in split['train_group_ids'])
    val_gids   = set(str(x) for x in split['val_group_ids'])
    test_gids  = set(str(x) for x in split['test_group_ids'])

    # Verify disjointness
    assert not (train_gids & val_gids),  "Train-Val overlap!"
    assert not (train_gids & test_gids), "Train-Test overlap!"
    assert not (val_gids  & test_gids),  "Val-Test overlap!"

    group_map = {}
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
    tr_cnt = Counter(t['label'] for t in train_trajs)
    va_cnt = Counter(t['label'] for t in val_trajs)
    te_cnt = Counter(t['label'] for t in test_trajs)

    # 2. Quality-control print
    print('\n' + '='*65)
    print('  TRACEGUARD Transformer v4.1  —  CLS-Token Pooling')
    print('='*65)
    print(f'  Dataset         : {data_path}')
    print(f'  Total trajs     : {len(trajs)}')
    print(f'  Groups          : 200')
    print(f'  Classes (global): {dict(Counter(t["label"] for t in trajs))}')
    print(f'  Train groups    : {len(train_gids)}  trajs={len(train_trajs)}  {dict(tr_cnt)}')
    print(f'  Val   groups    : {len(val_gids)}   trajs={len(val_trajs)}   {dict(va_cnt)}')
    print(f'  Test  groups    : {len(test_gids)}   trajs={len(test_trajs)}   {dict(te_cnt)}')
    print(f'  Embedding model : {EMB_MODEL_NAME}  dim={EMB_DIM}')
    print(f'  Architecture    : CLS-Transformer  model_dim={MODEL_DIM}  layers={N_LAYERS}  heads={N_HEADS}  ff={FF_DIM}  drop={DROPOUT}')
    print(f'  Optimizer       : AdamW  lr={LR}  wd={WEIGHT_DECAY}')
    print(f'  Batch size      : {BATCH_SIZE}  max_epochs={MAX_EPOCHS}  patience={PATIENCE}')
    print(f'  Seed            : {seed}')
    print(f'  Threshold       : {THRESHOLD}')
    print('='*65 + '\n')

    # 3. Embeddings
    emb_dict = build_emb_dict(trajs, seed)
    # Save meta
    with open(RESULTS / 'embeddings' / f'embeddings_meta_seed{seed}.json', 'w') as f:
        json.dump({'model': EMB_MODEL_NAME, 'n_trajectories': len(emb_dict)}, f)

    # 4. DataLoaders
    train_loader = make_loader(train_trajs, emb_dict, shuffle=True,  seed=seed)
    val_loader   = make_loader(val_trajs,   emb_dict, shuffle=False, seed=seed)
    test_loader  = make_loader(test_trajs,  emb_dict, shuffle=False, seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # 5. Model
    model = CLSTransformer(
        input_dim=EMB_DIM, model_dim=MODEL_DIM, num_heads=N_HEADS,
        ff_dim=FF_DIM, num_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] CLS-Transformer  trainable params: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 6. Training loop
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    history     = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    print('\n[Training]')
    for epoch in range(1, MAX_EPOCHS + 1):
        # — Train —
        model.train()
        train_loss = 0.0
        for embs, labels, lengths, _ in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embs, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * embs.size(0)
        train_loss /= len(train_loader.dataset)

        # — Validate —
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embs, labels, lengths, _ in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs, lengths)
                val_loss += criterion(logits, labels).item() * embs.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_acc  = float((np.array(all_preds) == np.array(all_labels)).mean())

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)

        improved = val_f1 > best_val_f1
        marker   = ' *' if improved else ''
        # Per-class val breakdown for monitoring
        p_b  = all_preds.count(0) if hasattr(all_preds,'count') else sum(1 for p in all_preds if p==0)
        p_ir = sum(1 for p in all_preds if p==1)
        p_h  = sum(1 for p in all_preds if p==2)
        print(f'  Ep {epoch:03d} | TrLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | '
              f'ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}{marker} | '
              f'pred[B={p_b} IR={p_ir} H={p_h}]')

        if improved:
            best_val_f1 = val_f1
            best_state  = {'model_state_dict': model.state_dict(), 'best_val_f1': best_val_f1}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  Early stopping at epoch {epoch}  (patience={PATIENCE})')
                break

    # 7. Load best model
    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])

    tag = f'transformer_seed{seed}'
    ckpt_path = RESULTS / 'checkpoints' / f'{tag}_best.pth'
    torch.save(best_state, ckpt_path)
    with open(RESULTS / 'metrics' / f'{tag}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\n[Checkpoint] {ckpt_path}  (best_val_f1={best_val_f1:.4f})')

    # 8. Full test evaluation
    model.eval()
    y_true, y_pred, y_prob, tids = [], [], [], []
    with torch.no_grad():
        for embs, labels, lengths, traj_ids in test_loader:
            embs = embs.to(device)
            logits = model(embs, lengths)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
            tids.extend(traj_ids)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    with open(RESULTS / 'metrics' / f'{tag}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_cm(y_true, y_pred, tag)
    save_predictions(y_true, y_pred, y_prob, tids, tag)

    # ── REQUIRED SUCCESS CHECK ─────────────────────────────────────────────────
    print('\n' + '='*65)
    print('  SEED-42 CLASSIFICATION REPORT')
    print('='*65)
    print(f'  Accuracy        : {metrics["accuracy"]:.4f}')
    print(f'  Macro Precision : {metrics["macro_precision"]:.4f}')
    print(f'  Macro Recall    : {metrics["macro_recall"]:.4f}')
    print(f'  Macro F1        : {metrics["macro_f1"]:.4f}')
    print()
    for cls in CLASS_NAMES:
        pc = metrics['per_class'][cls]
        au = pc['auroc']
        pr = pc['pr_auc']
        print(f'  {cls:22s}  P={pc["precision"]:.3f}  R={pc["recall"]:.3f}  '
              f'F1={pc["f1"]:.3f}  AUROC={au if au is None else f"{au:.3f}"}  '
              f'PR-AUC={pr if pr is None else f"{pr:.3f}"}')
    print()

    inj_f1 = metrics['per_class']['INJECTION_RESISTED']['f1']
    if inj_f1 <= 0.0:
        print('  [STOP] INJECTION_RESISTED F1 = 0.0')
        print('  The class collapse persists. DO NOT proceed to prefix evaluation.')
        print('  Stopping as required. Diagnostics needed.')
        print('='*65)
        return metrics

    print(f'  [OK] INJECTION_RESISTED F1 = {inj_f1:.4f}  > 0  — proceeding to prefix evaluation.')
    print('='*65)

    # 9. Prefix evaluation
    print('\n[Prefix Evaluation]')
    prefix_records = []
    for t in test_trajs:
        tid       = t['trajectory_id']
        true_lbl  = t['label']
        dev_step  = t.get('deviation_step')
        embs      = emb_dict[tid]          # (T, 384)
        seq_len   = len(embs)
        for k in range(1, seq_len + 1):
            prefix = embs[:k]              # (k, 384)
            with torch.no_grad():
                tensor = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(tensor, [k])
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_cls = INV_LABEL_MAP[int(probs.argmax())]
            prefix_records.append({
                'trajectory_id'    : tid,
                'prefix_length'    : k,
                'true_label'       : true_lbl,
                'predicted_label'  : pred_cls,
                'p_benign'         : float(probs[0]),
                'p_injection_resisted': float(probs[1]),
                'p_hijacked'       : float(probs[2]),
                'deviation_step'   : dev_step,
            })

    df_prefix = pd.DataFrame(prefix_records)
    pfx_path  = RESULTS / 'prefix_detection' / f'{tag}_prefixes.csv'
    df_prefix.to_csv(pfx_path, index=False)
    print(f'  Prefix records: {len(df_prefix)}  saved to {pfx_path}')

    # 10. Early detection (HIJACKED, threshold=0.5)
    print('\n[Early Detection]')
    hijacked = df_prefix[df_prefix['true_label'] == 'HIJACKED']
    det_records = []
    for tid, grp in hijacked.groupby('trajectory_id'):
        grp     = grp.sort_values('prefix_length')
        dev_s   = grp['deviation_step'].iloc[0]
        detected_rows = grp[grp['p_hijacked'] >= THRESHOLD]
        if not detected_rows.empty:
            det_step   = int(detected_rows['prefix_length'].iloc[0])
            det_flag   = True
        else:
            det_step   = int(grp['prefix_length'].max()) + 1
            det_flag   = False
        ref        = dev_s if dev_s is not None else grp['prefix_length'].max() + 1
        latency    = det_step - ref
        pre_action = det_step < ref
        det_records.append({
            'trajectory_id': tid,
            'deviation_step': dev_s,
            'detection_step': det_step,
            'detected': det_flag,
            'latency': latency,
            'pre_action': pre_action,
        })

    df_det  = pd.DataFrame(det_records)
    det_path = RESULTS / 'latency' / f'{tag}_early_detection.csv'
    df_det.to_csv(det_path, index=False)

    n_det        = int(df_det['detected'].sum())
    n_total_h    = len(df_det)
    n_pre        = int(df_det['pre_action'].sum())
    n_never      = n_total_h - n_det
    lat_detected = df_det.loc[df_det['detected'] == True, 'latency']

    print(f'  Detection rate  : {n_det}/{n_total_h} = {n_det/n_total_h:.3f}')
    print(f'  Pre-action rate : {n_pre}/{n_total_h} = {n_pre/n_total_h:.3f}')
    print(f'  Never detected  : {n_never}')
    if len(lat_detected) > 0:
        print(f'  Latency  mean   : {lat_detected.mean():.3f}')
        print(f'  Latency  median : {lat_detected.median():.3f}')
        print(f'  Latency  std    : {lat_detected.std():.3f}')
        print(f'  Latency  min    : {lat_detected.min():.3f}')
        print(f'  Latency  max    : {lat_detected.max():.3f}')

    # 11. Save experiment config
    with open(RESULTS / 'configs' / f'experiment_config_seed{seed}.json', 'w') as f:
        json.dump({
            'dataset'    : str(data_path),
            'dataset_size': len(trajs),
            'split_path' : str(split_path),
            'seed'       : seed,
            'embedding_model': EMB_MODEL_NAME,
            'embedding_dim'  : EMB_DIM,
            'architecture'   : {
                'type': 'CLS-Transformer',
                'pooling': 'CLS-token',
                'input_dim': EMB_DIM,
                'model_dim': MODEL_DIM,
                'layers': N_LAYERS,
                'heads' : N_HEADS,
                'ff_dim': FF_DIM,
                'dropout': DROPOUT,
                'num_classes': NUM_CLASSES,
            },
            'optimizer': 'AdamW',
            'learning_rate' : LR,
            'weight_decay'  : WEIGHT_DECAY,
            'batch_size'    : BATCH_SIZE,
            'max_epochs'    : MAX_EPOCHS,
            'patience'      : PATIENCE,
            'detection_threshold': THRESHOLD,
            'model_selection': 'val_macro_f1',
        }, f, indent=2)

    with open(RESULTS / 'configs' / f'split_seed{seed}.json', 'w') as f:
        import shutil
        shutil.copy(split_path, RESULTS / 'configs' / f'split_seed{seed}.json')

    print(f'\n[DONE] Seed-{seed} experiment complete.')
    print(f'  Results in: {RESULTS}')
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(seed=args.seed)
