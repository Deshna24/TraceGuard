"""
run_full_transformer_and_baselines.py

Complete pipeline for TRACEGUARD Transformer Baseline (v4.1):
1. 5-Seed Robustness Evaluation (seeds 42, 123, 456, 789, 1011)
2. Step-Order Shuffling Ablation (seed 42)
3. Non-Sequential Baseline (Mean Pooling + Logistic Regression)
4. Final Model Comparison Table (Mean Pooling + LogReg vs LSTM vs Transformer)
"""

import os, json, sys, shutil, random
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
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

RESULTS_TF = ROOT / 'results' / 'transformer_v4_1'
RESULTS_COMP = ROOT / 'results' / 'final_comparison'

for d in ['checkpoints','metrics','plots','confusion_matrices','prefix_detection',
          'latency','ablations','robustness','splits','embeddings','configs','reports','handoff']:
    (RESULTS_TF / d).mkdir(parents=True, exist_ok=True)
RESULTS_COMP.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters & Config ──────────────────────────────────────────────────
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
SEEDS           = [42, 123, 456, 789, 1011]

DATA_PATH = ROOT / 'data' / 'raw' / 'trajectories' / 'traceguard_v4_1.jsonl'
if not DATA_PATH.exists():
    DATA_PATH = ROOT / 'traceguard' / 'data' / 'traceguard_v4_1.jsonl'

EMB_CACHE_PATH = ROOT / 'traceguard' / 'outputs' / 'embeddings_cache' / 'embeddings_all-MiniLM-L6-v2.npy'
if not EMB_CACHE_PATH.exists():
    EMB_CACHE_PATH = RESULTS_TF / 'embeddings' / f'embeddings_{EMB_MODEL_NAME.replace("/","_")}.npy'


# ── Positional Encoding & Transformer Model ───────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class CLSTransformer(nn.Module):
    def __init__(self, input_dim=384, model_dim=128, num_heads=4,
                 ff_dim=256, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.model_dim   = model_dim
        self.input_proj  = nn.Linear(input_dim, model_dim)
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
        B, T, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)

        device = x.device
        mask = torch.zeros(B, T + 1, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            if L < T:
                mask[i, L + 1:] = True

        out = self.transformer_encoder(x, src_key_padding_mask=mask)
        cls_out = out[:, 0, :]
        cls_out = self.cls_dropout(cls_out)
        return self.classifier(cls_out)


# ── Data Utilities ────────────────────────────────────────────────────────────
def load_dataset():
    trajs = []
    with open(DATA_PATH, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))
    return trajs

def load_split_file(seed):
    split_paths = [
        ROOT / 'traceguard' / 'outputs' / 'splits' / f'split_seed{seed}.json',
        ROOT / 'traceguard' / 'outputs' / 'handoff' / 'splits' / f'split_seed{seed}.json',
    ]
    sp = next((p for p in split_paths if p.exists()), None)
    assert sp is not None, f"Split file for seed {seed} not found!"
    with open(sp) as f:
        return json.load(f)

def load_embeddings():
    assert EMB_CACHE_PATH.exists(), f"Embeddings cache {EMB_CACHE_PATH} not found!"
    return np.load(EMB_CACHE_PATH, allow_pickle=True).item()

class TrajectoryDataset(Dataset):
    def __init__(self, trajs, emb_dict):
        self.trajs    = trajs
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        t     = self.trajs[idx]
        tid   = t['trajectory_id']
        emb   = self.emb_dict[tid]
        label = LABEL_MAP[t['label']]
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
    ax.set_title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    out = RESULTS_TF / 'confusion_matrices' / f'{tag}_confusion_matrix.png'
    plt.savefig(out, dpi=150); plt.close()
    return out


# ── Train Single Transformer Seed ─────────────────────────────────────────────
def train_eval_transformer_seed(seed, trajs, emb_dict):
    print(f'\n' + '='*65)
    print(f'  RUNNING TRANSFORMER SEED {seed}')
    print('='*65)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = load_split_file(seed)
    train_gids = set(str(x) for x in split['train_group_ids'])
    val_gids   = set(str(x) for x in split['val_group_ids'])
    test_gids  = set(str(x) for x in split['test_group_ids'])

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

    ds_tr  = TrajectoryDataset(train_trajs, emb_dict)
    ds_val = TrajectoryDataset(val_trajs, emb_dict)
    ds_te  = TrajectoryDataset(test_trajs, emb_dict)

    g_tr = torch.Generator(); g_tr.manual_seed(seed)
    loader_tr  = DataLoader(ds_tr,  batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, generator=g_tr)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    loader_te  = DataLoader(ds_te,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = CLSTransformer(input_dim=EMB_DIM, model_dim=MODEL_DIM, num_heads=N_HEADS,
                            ff_dim=FF_DIM, num_layers=N_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    history     = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_macro_f1': []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for embs, labels, lengths, _ in loader_tr:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(embs, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * embs.size(0)
        tr_loss /= len(loader_tr.dataset)

        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for embs, labels, lengths, _ in loader_val:
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs, lengths)
                val_loss += criterion(logits, labels).item() * embs.size(0)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(loader_val.dataset)
        val_f1   = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_acc  = float((np.array(val_preds) == np.array(val_labels)).mean())

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {'model_state_dict': model.state_dict(), 'best_val_f1': best_val_f1}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  Seed {seed} early stopping at epoch {epoch} (best_val_f1={best_val_f1:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])

    tag = f'transformer_seed{seed}'
    torch.save(best_state, RESULTS_TF / 'checkpoints' / f'{tag}_best.pth')
    with open(RESULTS_TF / 'metrics' / f'{tag}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Test Evaluation
    model.eval()
    y_true, y_pred, y_prob, tids = [], [], [], []
    with torch.no_grad():
        for embs, labels, lengths, traj_ids in loader_te:
            embs = embs.to(device)
            logits = model(embs, lengths)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
            tids.extend(traj_ids)

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    with open(RESULTS_TF / 'metrics' / f'{tag}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_cm(y_true, y_pred, tag)

    # Prefix Evaluation
    prefix_records = []
    for t in test_trajs:
        tid      = t['trajectory_id']
        true_lbl = t['label']
        dev_step = t.get('deviation_step')
        embs     = emb_dict[tid]
        for k in range(1, len(embs) + 1):
            prefix = embs[:k]
            with torch.no_grad():
                tensor = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(tensor, [k])
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_cls = INV_LABEL_MAP[int(probs.argmax())]
            prefix_records.append({
                'trajectory_id': tid, 'prefix_length': k, 'true_label': true_lbl,
                'predicted_label': pred_cls,
                'p_benign': float(probs[0]), 'p_injection_resisted': float(probs[1]), 'p_hijacked': float(probs[2]),
                'deviation_step': dev_step,
            })
    df_prefix = pd.DataFrame(prefix_records)
    df_prefix.to_csv(RESULTS_TF / 'prefix_detection' / f'{tag}_prefixes.csv', index=False)

    # Early Detection
    hijacked = df_prefix[df_prefix['true_label'] == 'HIJACKED']
    det_records = []
    for tid_h, grp in hijacked.groupby('trajectory_id'):
        grp   = grp.sort_values('prefix_length')
        dev_s = grp['deviation_step'].iloc[0]
        det_rows = grp[grp['p_hijacked'] >= THRESHOLD]
        if not det_rows.empty:
            det_step = int(det_rows['prefix_length'].iloc[0])
            det_flag = True
        else:
            det_step = int(grp['prefix_length'].max()) + 1
            det_flag = False
        ref     = dev_s if dev_s is not None else grp['prefix_length'].max() + 1
        lat     = det_step - ref
        pre_act = det_step < ref
        det_records.append({
            'trajectory_id': tid_h, 'deviation_step': dev_s, 'detection_step': det_step,
            'detected': det_flag, 'latency': lat, 'pre_action': pre_act,
        })
    df_det = pd.DataFrame(det_records)
    df_det.to_csv(RESULTS_TF / 'latency' / f'{tag}_early_detection.csv', index=False)

    pre_rate = float(df_det['pre_action'].mean())
    lat_det  = df_det.loc[df_det['detected'] == True, 'latency']
    mean_lat = float(lat_det.mean()) if len(lat_det) > 0 else float('nan')

    print(f'  Seed {seed} Test Results: Acc={metrics["accuracy"]:.4f} | MacroF1={metrics["macro_f1"]:.4f} | '
          f'BENIGN_F1={metrics["per_class"]["BENIGN"]["f1"]:.4f} | '
          f'RESISTED_F1={metrics["per_class"]["INJECTION_RESISTED"]["f1"]:.4f} | '
          f'HIJACKED_F1={metrics["per_class"]["HIJACKED"]["f1"]:.4f} | '
          f'PreActionRate={pre_rate:.4f}')

    return {
        'seed': seed,
        'accuracy': metrics['accuracy'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
        'benign_f1': metrics['per_class']['BENIGN']['f1'],
        'resisted_f1': metrics['per_class']['INJECTION_RESISTED']['f1'],
        'hijacked_f1': metrics['per_class']['HIJACKED']['f1'],
        'hijacked_precision': metrics['per_class']['HIJACKED']['precision'],
        'hijacked_recall': metrics['per_class']['HIJACKED']['recall'],
        'hijacked_auroc': metrics['per_class']['HIJACKED']['auroc'],
        'pre_action_rate': pre_rate,
        'mean_latency': mean_lat,
    }, model


# ── Step-Order Shuffling Ablation (Seed 42) ───────────────────────────────────
def run_step_shuffling_ablation(model, trajs, emb_dict, seed=42, n_reps=10):
    print('\n' + '='*65)
    print(f'  RUNNING STEP-ORDER SHUFFLING ABLATION (SEED {seed}, N={n_reps})')
    print('='*65)
    
    split = load_split_file(seed)
    test_gids = set(str(x) for x in split['test_group_ids'])
    group_map = {}
    for t in trajs:
        group_map.setdefault(str(t['counterfactual_group_id']), []).append(t)
    test_trajs = []
    for gid in test_gids:
        test_trajs.extend(group_map.get(gid, []))

    device = next(model.parameters()).device
    model.eval()

    orig_y_true, orig_y_pred = [], []
    for t in test_trajs:
        tid   = t['trajectory_id']
        emb   = emb_dict[tid]
        label = LABEL_MAP[t['label']]
        with torch.no_grad():
            tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(tensor, [len(emb)])
            pred   = int(logits.argmax(dim=1)[0].cpu())
        orig_y_true.append(label)
        orig_y_pred.append(pred)

    orig_acc = float(accuracy_score(orig_y_true, orig_y_pred))
    orig_f1  = float(f1_score(orig_y_true, orig_y_pred, average='macro', zero_division=0))

    shuffled_accs, shuffled_f1s = [], []
    for rep in range(n_reps):
        np.random.seed(seed + rep * 100)
        shuf_y_pred = []
        for t in test_trajs:
            tid  = t['trajectory_id']
            emb  = emb_dict[tid].copy()
            idx  = np.random.permutation(len(emb))
            shuf_emb = emb[idx]
            with torch.no_grad():
                tensor = torch.tensor(shuf_emb, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(tensor, [len(shuf_emb)])
                pred   = int(logits.argmax(dim=1)[0].cpu())
            shuf_y_pred.append(pred)
        
        acc_rep = float(accuracy_score(orig_y_true, shuf_y_pred))
        f1_rep  = float(f1_score(orig_y_true, shuf_y_pred, average='macro', zero_division=0))
        shuffled_accs.append(acc_rep)
        shuffled_f1s.append(f1_rep)

    shuf_acc_mean = float(np.mean(shuffled_accs))
    shuf_acc_std  = float(np.std(shuffled_accs))
    shuf_f1_mean  = float(np.mean(shuffled_f1s))
    shuf_f1_std   = float(np.std(shuffled_f1s))
    f1_delta      = orig_f1 - shuf_f1_mean

    res = {
        'seed': seed,
        'n_repetitions': n_reps,
        'original_accuracy': orig_acc,
        'original_macro_f1': orig_f1,
        'shuffled_accuracy_mean': shuf_acc_mean,
        'shuffled_accuracy_std': shuf_acc_std,
        'shuffled_macro_f1_mean': shuf_f1_mean,
        'shuffled_macro_f1_std': shuf_f1_std,
        'f1_delta': f1_delta,
    }

    with open(RESULTS_TF / 'ablations' / 'step_shuffling_ablation.json', 'w') as f:
        json.dump(res, f, indent=2)
    pd.DataFrame([res]).to_csv(RESULTS_TF / 'ablations' / 'step_shuffling_ablation.csv', index=False)

    print(f'  Original  : Acc={orig_acc:.4f} | MacroF1={orig_f1:.4f}')
    print(f'  Shuffled  : Acc={shuf_acc_mean:.4f}±{shuf_acc_std:.4f} | MacroF1={shuf_f1_mean:.4f}±{shuf_f1_std:.4f}')
    print(f'  F1 Delta  : {f1_delta:.4f}')
    return res


# ── Non-Sequential Baseline (Mean Pooling + Logistic Regression) ─────────────
def run_non_sequential_baseline(trajs, emb_dict, seed=42):
    print('\n' + '='*65)
    print(f'  RUNNING NON-SEQUENTIAL BASELINE (Mean Pooling + LogReg, Seed {seed})')
    print('='*65)

    split = load_split_file(seed)
    train_gids = set(str(x) for x in split['train_group_ids'])
    val_gids   = set(str(x) for x in split['val_group_ids'])
    test_gids  = set(str(x) for x in split['test_group_ids'])

    group_map = {}
    for t in trajs:
        group_map.setdefault(str(t['counterfactual_group_id']), []).append(t)

    def extract_features(gids):
        X, y, tids = [], [], []
        for gid in gids:
            for t in group_map.get(gid, []):
                tid = t['trajectory_id']
                emb = emb_dict[tid]
                X.append(emb.mean(axis=0))  # Mean pool across step dimension
                y.append(LABEL_MAP[t['label']])
                tids.append(tid)
        return np.array(X), np.array(y), tids

    X_train, y_train, _ = extract_features(train_gids)
    X_val,   y_val,   _ = extract_features(val_gids)
    X_test,  y_test, tids_test = extract_features(test_gids)

    # Train Logistic Regression
    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Prefix Evaluation for Baseline
    test_trajs = []
    for gid in test_gids:
        test_trajs.extend(group_map.get(gid, []))

    prefix_records = []
    for t in test_trajs:
        tid      = t['trajectory_id']
        true_lbl = t['label']
        dev_step = t.get('deviation_step')
        embs     = emb_dict[tid]
        for k in range(1, len(embs) + 1):
            prefix_mean = embs[:k].mean(axis=0, keepdims=True)
            probs       = clf.predict_proba(prefix_mean)[0]
            pred_cls    = INV_LABEL_MAP[int(probs.argmax())]
            prefix_records.append({
                'trajectory_id': tid, 'prefix_length': k, 'true_label': true_lbl,
                'predicted_label': pred_cls,
                'p_benign': float(probs[0]), 'p_injection_resisted': float(probs[1]), 'p_hijacked': float(probs[2]),
                'deviation_step': dev_step,
            })
    df_prefix = pd.DataFrame(prefix_records)

    # Early Detection
    hijacked = df_prefix[df_prefix['true_label'] == 'HIJACKED']
    det_records = []
    for tid_h, grp in hijacked.groupby('trajectory_id'):
        grp   = grp.sort_values('prefix_length')
        dev_s = grp['deviation_step'].iloc[0]
        det_rows = grp[grp['p_hijacked'] >= THRESHOLD]
        if not det_rows.empty:
            det_step = int(det_rows['prefix_length'].iloc[0])
            det_flag = True
        else:
            det_step = int(grp['prefix_length'].max()) + 1
            det_flag = False
        ref     = dev_s if dev_s is not None else grp['prefix_length'].max() + 1
        lat     = det_step - ref
        pre_act = det_step < ref
        det_records.append({
            'trajectory_id': tid_h, 'deviation_step': dev_s, 'detection_step': det_step,
            'detected': det_flag, 'latency': lat, 'pre_action': pre_act,
        })
    df_det = pd.DataFrame(det_records)
    pre_rate = float(df_det['pre_action'].mean())
    lat_det  = df_det.loc[df_det['detected'] == True, 'latency']
    mean_lat = float(lat_det.mean()) if len(lat_det) > 0 else float('nan')

    print(f'  Baseline Test Results: Acc={metrics["accuracy"]:.4f} | MacroF1={metrics["macro_f1"]:.4f} | '
          f'HIJACKED_F1={metrics["per_class"]["HIJACKED"]["f1"]:.4f} | '
          f'PreActionRate={pre_rate:.4f} | MeanLat={mean_lat:.4f}')

    return {
        'model': 'Mean Pooling + Logistic Regression',
        'accuracy': metrics['accuracy'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'macro_f1': metrics['macro_f1'],
        'hijacked_precision': metrics['per_class']['HIJACKED']['precision'],
        'hijacked_recall': metrics['per_class']['HIJACKED']['recall'],
        'hijacked_f1': metrics['per_class']['HIJACKED']['f1'],
        'hijacked_auroc': metrics['per_class']['HIJACKED']['auroc'],
        'pre_action_rate': pre_rate,
        'mean_latency': mean_lat,
    }


# ── MAIN PIPELINE EXECUTION ───────────────────────────────────────────────────
def main():
    print('\n=================================================================')
    print('  TRACEGUARD TRANSFORMER & BASELINE EXPERIMENT SUITE v4.1')
    print('=================================================================\n')

    trajs    = load_dataset()
    emb_dict = load_embeddings()

    # 1. Five-Seed Transformer Evaluation
    tf_seed_results = []
    seed42_model    = None

    for seed in SEEDS:
        res, model = train_eval_transformer_seed(seed, trajs, emb_dict)
        tf_seed_results.append(res)
        if seed == 42:
            seed42_model = model

    # Calculate 5-Seed Aggregates
    metrics_to_agg = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                      'benign_f1', 'resisted_f1', 'hijacked_f1', 'pre_action_rate']
    tf_agg = {}
    for m in metrics_to_agg:
        vals = [r[m] for r in tf_seed_results]
        tf_agg[m] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals))
        }

    robustness_data = {
        'per_seed': tf_seed_results,
        'aggregate': tf_agg
    }

    with open(RESULTS_TF / 'robustness' / 'five_seed_robustness.json', 'w') as f:
        json.dump(robustness_data, f, indent=2)
    pd.DataFrame(tf_seed_results).to_csv(RESULTS_TF / 'robustness' / 'five_seed_robustness.csv', index=False)

    print('\n[FIVE-SEED ROBUSTNESS SUMMARY — TRANSFORMER CLS]')
    for m in metrics_to_agg:
        print(f'  {m:20s}: {tf_agg[m]["mean"]:.4f} ± {tf_agg[m]["std"]:.4f}')

    # 2. Step-Order Shuffling Ablation (Seed 42)
    ablation_res = run_step_shuffling_ablation(seed42_model, trajs, emb_dict, seed=42, n_reps=10)

    # 3. Non-Sequential Baseline Evaluation (Seed 42)
    baseline_res = run_non_sequential_baseline(trajs, emb_dict, seed=42)

    # 4. Load Frozen LSTM Results (Seed 42)
    lstm_metrics_path = ROOT / 'traceguard' / 'outputs' / 'metrics' / 'lstm_seed42_metrics.json'
    lstm_lat_path     = ROOT / 'traceguard' / 'outputs' / 'metrics' / 'latency_stats_lstm_seed42.json'
    assert lstm_metrics_path.exists() and lstm_lat_path.exists(), "Frozen LSTM results missing!"

    with open(lstm_metrics_path) as f:
        lstm_m = json.load(f)
    with open(lstm_lat_path) as f:
        lstm_l = json.load(f)

    lstm_res = {
        'model': 'LSTM',
        'accuracy': lstm_m['accuracy'],
        'macro_precision': lstm_m['macro_precision'],
        'macro_recall': lstm_m['macro_recall'],
        'macro_f1': lstm_m['macro_f1'],
        'hijacked_precision': lstm_m['per_class']['HIJACKED']['precision'],
        'hijacked_recall': lstm_m['per_class']['HIJACKED']['recall'],
        'hijacked_f1': lstm_m['per_class']['HIJACKED']['f1'],
        'hijacked_auroc': lstm_m['per_class']['HIJACKED']['auroc'],
        'pre_action_rate': lstm_l['pre_action_detection_rate'],
        'mean_latency': lstm_l['mean_latency'],
    }

    # Extract Transformer Seed 42 for comparison table
    tf_s42_res = next(r for r in tf_seed_results if r['seed'] == 42)
    tf_comp_res = {
        'model': 'Transformer (CLS)',
        'accuracy': tf_s42_res['accuracy'],
        'macro_precision': tf_s42_res['macro_precision'],
        'macro_recall': tf_s42_res['macro_recall'],
        'macro_f1': tf_s42_res['macro_f1'],
        'hijacked_precision': tf_s42_res['hijacked_precision'],
        'hijacked_recall': tf_s42_res['hijacked_recall'],
        'hijacked_f1': tf_s42_res['hijacked_f1'],
        'hijacked_auroc': tf_s42_res['hijacked_auroc'],
        'pre_action_rate': tf_s42_res['pre_action_rate'],
        'mean_latency': tf_s42_res['mean_latency'],
    }

    # 5. Build Final Comparison Table
    comparison_list = [baseline_res, lstm_res, tf_comp_res]
    df_comp = pd.DataFrame(comparison_list)

    # Reorder columns for clean presentation
    cols = ['model', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'hijacked_precision', 'hijacked_recall', 'hijacked_f1', 'hijacked_auroc',
            'pre_action_rate', 'mean_latency']
    df_comp = df_comp[cols]

    # Save comparison artifacts
    df_comp.to_csv(RESULTS_COMP / 'model_comparison.csv', index=False)
    with open(RESULTS_COMP / 'model_comparison.json', 'w') as f:
        json.dump(comparison_list, f, indent=2)

    # Also save inside results/transformer_v4_1 for self-containment
    df_comp.to_csv(RESULTS_TF / 'model_comparison.csv', index=False)

    md_table = (
        "# TRACEGUARD Baseline & Model Comparison Table (v4.1 — Seed 42)\n\n"
        "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | HIJACKED Precision | HIJACKED Recall | HIJACKED F1 | HIJACKED AUROC | Pre-action Detection Rate | Mean Detection Latency |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    )
    for r in comparison_list:
        lat_str = f"{r['mean_latency']:.3f}" if not np.isnan(r['mean_latency']) else "N/A"
        md_table += (
            f"| **{r['model']}** | {r['accuracy']:.4f} | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['hijacked_precision']:.4f} | {r['hijacked_recall']:.4f} | "
            f"{r['hijacked_f1']:.4f} | {r['hijacked_auroc']:.4f} | {r['pre_action_rate']:.4f} | {lat_str} |\n"
        )

    with open(RESULTS_COMP / 'model_comparison.md', 'w') as f:
        f.write(md_table)

    print('\n' + '='*65)
    print('  FINAL MODEL COMPARISON TABLE (SEED 42)')
    print('='*65)
    print(md_table)
    print('='*65)
    print(f'[COMPLETE] All artifacts saved to {RESULTS_TF} and {RESULTS_COMP}')


if __name__ == '__main__':
    main()
