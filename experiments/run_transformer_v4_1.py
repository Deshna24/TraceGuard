"""
run_transformer_v4_1.py - Transformer baseline experiment for TRACEGUARD v4.1

Follows the v4.1 experiment spec: uses frozen dataset,
uses the existing split_seed42.json, embeddings from all-MiniLM-L6-v2,
and writes outputs under results/transformer_v4_1/.

Usage:
    python experiments/run_transformer_v4_1.py --seed 42
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse repo utilities
import sys
sys.path.insert(0, os.path.abspath('.'))
# Ensure top-level `src` directory is importable (some modules import `src.*`)
sys.path.insert(0, str(Path('.').resolve() / 'src'))
from traceguard.src.config import DATA_PATH, SPLITS_DIR, EMBEDDING_MODEL_NAME, DETECTION_THRESHOLD, PRIMARY_SEED
# Import constants for validation
from traceguard.src.config import TOTAL_TRAJECTORIES, TOTAL_GROUPS, STEPS_PER_TRAJECTORY, CLASS_NAMES

# Delay importing traceguard.src.data_loader (it imports `src.config`) until runtime
load_dataset = None
get_group_map = None

# Create results structure for transformer experiment
RESULTS_ROOT = Path('results') / 'transformer_v4_1'
SUBDIRS = [
    'checkpoints','metrics','plots','confusion_matrices','prefix_detection',
    'latency','ablations','robustness','splits','embeddings','configs','reports','handoff'
]
for d in SUBDIRS:
    (RESULTS_ROOT / d).mkdir(parents=True, exist_ok=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)


class LightweightTransformer(nn.Module):
    def __init__(self, input_dim=384, model_dim=128, num_heads=4, ff_dim=256, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        # src_key_padding_mask expects shape [B, T] with True in positions that are pads
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= torch.tensor(lengths, device=x.device).unsqueeze(1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=mask)
        pooled = []
        for i, L in enumerate(lengths):
            pooled.append(out[i, :L].mean(dim=0))
        pooled = torch.stack(pooled)
        pooled = self.pooling_dropout(pooled)
        return self.classifier(pooled)


def load_split_json(seed):
    path = Path(SPLITS_DIR) / f"split_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Required split not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset_local(path=None):
    p = Path(path) if path else DATA_PATH
    print(f"\n[DataLoader] Loading dataset from: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {p}")
    trajs = []
    with open(p, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            trajs.append(json.loads(line))
    # Basic validations (mirror traceguard checks)
    if len(trajs) != TOTAL_TRAJECTORIES:
        raise AssertionError(f"Expected {TOTAL_TRAJECTORIES} trajectories, found {len(trajs)}")
    # required fields
    required = ["trajectory_id", "counterfactual_group_id", "label", "steps"]
    for t in trajs:
        for field in required:
            if field not in t:
                raise AssertionError(f"Missing field {field} in trajectory {t.get('trajectory_id')}")
    # label counts
    from collections import Counter
    label_counts = Counter(t["label"] for t in trajs)
    for cls in CLASS_NAMES:
        if label_counts.get(cls, 0) != TOTAL_TRAJECTORIES // len(CLASS_NAMES):
            # keep a warning rather than stop to be slightly tolerant
            print(f"[Warning] Expected balanced classes; found {label_counts}")
            break
    return trajs

def get_group_map_local(trajs):
    group_map = {}
    for t in trajs:
        gid = t["counterfactual_group_id"]
        group_map.setdefault(str(gid), []).append(t)
    return group_map


def make_dataloader(trajs, emb_dict, batch_size=32, shuffle=False, seed=42):
    # Local Dataset & collate to avoid importing traceguard.src.model_lstm (which depends on src package imports)
    from torch.utils.data import Dataset, DataLoader

    class TrajectoryDatasetLocal(Dataset):
        def __init__(self, trajectories, embeddings_dict):
            self.trajectories = trajectories
            self.embeddings_dict = embeddings_dict

        def __len__(self):
            return len(self.trajectories)

        def __getitem__(self, idx):
            traj = self.trajectories[idx]
            traj_id = traj['trajectory_id']
            emb = self.embeddings_dict[traj_id]
            label = traj['label']
            # convert label to int using traceguard mapping (avoid importing src.config here)
            lm = {'BENIGN': 0, 'INJECTION_RESISTED': 1, 'HIJACKED': 2}
            return torch.tensor(emb, dtype=torch.float32), torch.tensor(lm[label], dtype=torch.long), traj_id

    def collate_fn_local(batch):
        embs, labels, traj_ids = zip(*batch)
        lengths = [e.shape[0] for e in embs]
        max_len = max(lengths)
        dim = embs[0].shape[1]
        padded = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
        for i, e in enumerate(embs):
            padded[i, :e.shape[0], :] = e
        return padded, torch.stack(labels), lengths, traj_ids

    ds = TrajectoryDatasetLocal(trajs, emb_dict)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_local, generator=g if shuffle else None)


def run(seed=PRIMARY_SEED, force_embed=False):
    # 1. Load dataset (frozen)
    # Ensure src is on path for imports used by traceguard modules
    sys.path.insert(0, str(Path('.').resolve()))
    sys.path.insert(0, str(Path('.').resolve() / 'src'))
    # Prefer traceguard loader if importable, otherwise use local loader
    try:
        import importlib
        importlib.invalidate_caches()
        from traceguard.src.data_loader import load_dataset as _load_dataset, get_group_map as _get_group_map
        load_dataset_fn = _load_dataset
        get_group_map_fn = _get_group_map
    except Exception:
        load_dataset_fn = load_dataset_local
        get_group_map_fn = get_group_map_local

    trajectories = load_dataset_fn()
    dataset_size = len(trajectories)

    # 2. Load existing split
    split = load_split_json(seed)
    # verify disjoint groups
    train_ids = set(split['train_group_ids'])
    val_ids = set(split['val_group_ids'])
    test_ids = set(split['test_group_ids'])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError('Group overlap found in provided split')

    # Map groups to trajectories
    group_map = get_group_map_fn(trajectories)
    def gids_to_trajs(gids):
        out = []
        for gid in gids:
            out.extend(group_map[str(gid)])
        return out

    train_trajs = gids_to_trajs(split['train_group_ids'])
    val_trajs = gids_to_trajs(split['val_group_ids'])
    test_trajs = gids_to_trajs(split['test_group_ids'])

    # 3. Print configuration summary
    print('\n[CONFIGURATION SUMMARY]')
    print(f'  dataset path      : {DATA_PATH}')
    print(f'  dataset size      : {dataset_size}')
    print(f'  embedding model   : {EMBEDDING_MODEL_NAME}')
    print(f'  embedding dim     : 384')
    print(f'  train groups      : {len(split["train_group_ids"])}')
    print(f'  val groups        : {len(split["val_group_ids"])}')
    print(f'  test groups       : {len(split["test_group_ids"])}')
    print(f'  train trajectories: {len(train_trajs)}')
    print(f'  val trajectories  : {len(val_trajs)}')
    print(f'  test trajectories : {len(test_trajs)}')
    print('  Transformer arch   : model_dim=128, layers=2, heads=4, ff_dim=256, dropout=0.2')
    print(f'  random seed       : {seed}')
    print(f'  detection thresh  : {DETECTION_THRESHOLD}\n')

    # 4. Generate / load embeddings using traceguard's embedding utility (respects the frozen embedding model)
    try:
        from traceguard.src.embeddings import generate_embeddings as gen_emb
        emb_dict = gen_emb(trajectories)
    except Exception:
        # fallback to top-level src embeddings if present
        from src.embeddings import generate_embeddings as gen_emb2
        emb_dict = gen_emb2(trajectories, {"embedding_model": EMBEDDING_MODEL_NAME, "embeddings_dir": str(RESULTS_ROOT / 'embeddings')})

    # Save a copy of embeddings cache to results/transformer_v4_1/embeddings
    with open(RESULTS_ROOT / 'embeddings' / f'embeddings_meta_seed{seed}.json', 'w') as f:
        json.dump({'model': EMBEDDING_MODEL_NAME, 'n_trajectories': len(emb_dict)}, f)

    # 5. Build DataLoaders
    batch_size = 32
    train_loader = make_dataloader(train_trajs, emb_dict, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = make_dataloader(val_trajs, emb_dict, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = make_dataloader(test_trajs, emb_dict, batch_size=batch_size, shuffle=False, seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 6. Instantiate model
    model = LightweightTransformer(input_dim=384, model_dim=128, num_heads=4, ff_dim=256, num_layers=2, num_classes=3, dropout=0.2)
    model.to(device)

    # 7. Training: AdamW, early stopping on validation Macro F1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    patience = 7
    no_improve = 0
    max_epochs = 50

    from sklearn.metrics import f1_score

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    for epoch in range(1, max_epochs+1):
        model.train()
        train_loss = 0.0
        for embs, labels, lengths, _ in train_loader:
            embs = embs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(embs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * embs.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for embs, labels, lengths, _ in val_loader:
                embs = embs.to(device)
                labels = labels.to(device)
                logits = model(embs, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item() * embs.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(float(val_acc))
        history['val_macro_f1'].append(float(val_f1))

        improved = val_f1 > best_val_f1
        print(f"Epoch {epoch:03d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValAcc={val_acc:.4f} | ValMacroF1={val_f1:.4f}{' *' if improved else ''}")

        if improved:
            best_val_f1 = val_f1
            best_state = {'model': model.state_dict()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state['model'])

    # Save checkpoint and history
    ckpt_path = RESULTS_ROOT / 'checkpoints' / f"transformer_seed{seed}_best.pth"
    torch.save({'model_state_dict': model.state_dict(), 'best_val_f1': best_val_f1}, ckpt_path)
    with open(RESULTS_ROOT / 'metrics' / f'transformer_seed{seed}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Checkpoint saved: {ckpt_path}")

    # 8. Full evaluation on test set
    model.eval()
    y_true, y_pred, y_prob, tids = [], [], [], []
    with torch.no_grad():
        for embs, labels, lengths, traj_ids in test_loader:
            embs = embs.to(device)
            logits = model(embs, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
            tids.extend(traj_ids)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Local evaluation utilities (avoid importing traceguard.src.evaluate)
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, roc_auc_score, average_precision_score
    )
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    def compute_full_metrics_local(y_true, y_pred, y_prob):
        acc = float(accuracy_score(y_true, y_pred))
        prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
        prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

        auroc = {}
        pr_auc = {}
        for i, cls in enumerate(CLASS_NAMES):
            y_bin = (y_true == i).astype(int)
            if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                try:
                    auroc[cls] = float(roc_auc_score(y_bin, y_prob[:, i]))
                except Exception:
                    auroc[cls] = None
                try:
                    pr_auc[cls] = float(average_precision_score(y_bin, y_prob[:, i]))
                except Exception:
                    pr_auc[cls] = None
            else:
                auroc[cls] = None
                pr_auc[cls] = None

        per_class = {}
        for i, cls in enumerate(CLASS_NAMES):
            per_class[cls] = {
                'precision': float(prec_per[i]),
                'recall': float(rec_per[i]),
                'f1': float(f1_per[i]),
                'support': int(sup_per[i]),
                'auroc': auroc[cls],
                'pr_auc': pr_auc[cls],
            }

        metrics = {
            'accuracy': acc,
            'macro_precision': float(prec_mac),
            'macro_recall': float(rec_mac),
            'macro_f1': float(f1_mac),
            'per_class': per_class,
        }
        # save
        with open(RESULTS_ROOT / 'metrics' / f'transformer_seed{seed}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def plot_confusion_matrix_local(y_true, y_pred, tag):
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        fig, ax = plt.subplots(figsize=(7,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f"{tag} Confusion Matrix")
        plt.tight_layout()
        path = RESULTS_ROOT / 'confusion_matrices' / f"{tag}_confusion_matrix.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def save_predictions_local(y_true, y_pred, y_prob, traj_ids, tag):
        path = RESULTS_ROOT / 'metrics' / f"{tag}_predictions.csv"
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['trajectory_id', 'true_label', 'pred_label', 'p_benign', 'p_resisted', 'p_hijacked'])
            for tid, yt, yp, ypr in zip(traj_ids, y_true, y_pred, y_prob):
                writer.writerow([tid, CLASS_NAMES[int(yt)], CLASS_NAMES[int(yp)], f"{ypr[0]:.6f}", f"{ypr[1]:.6f}", f"{ypr[2]:.6f}"])
        return path

    metrics = compute_full_metrics_local(y_true, y_pred, y_prob)
    plot_confusion_matrix_local(y_true, y_pred, tag=f"transformer_seed{seed}")
    save_predictions_local(y_true, y_pred, y_prob, tids, tag=f"transformer_seed{seed}")

    # 9. Prefix evaluation
    # Local inverse label map
    INV_LABEL_MAP = {0: 'BENIGN', 1: 'INJECTION_RESISTED', 2: 'HIJACKED'}
    prefix_records = []
    for t in test_trajs:
        tid = t['trajectory_id']
        true_label = t['label']
        dev_step = t.get('deviation_step')
        embs = emb_dict[tid]
        seq_len = len(embs)
        for k in range(1, seq_len+1):
            prefix = embs[:k]
            with torch.no_grad():
                tensor = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(device)
                out = model(tensor, [k])
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_cls = INV_LABEL_MAP[int(probs.argmax())]
            prefix_records.append({
                'trajectory_id': tid,
                'prefix_length': k,
                'true_label': true_label,
                'predicted_label': pred_cls,
                'p_benign': float(probs[0]),
                'p_injection_resisted': float(probs[1]),
                'p_hijacked': float(probs[2]),
                'deviation_step': dev_step,
            })

    df_prefix = pd.DataFrame(prefix_records)
    df_prefix.to_csv(RESULTS_ROOT / 'prefix_detection' / f'transformer_seed{seed}_prefixes.csv', index=False)

    # 10. Early-detection metrics for HIJACKED (threshold 0.5)
    hijacked = df_prefix[df_prefix['true_label'] == 'HIJACKED']
    det_records = []
    for tid, grp in hijacked.groupby('trajectory_id'):
        grp = grp.sort_values('prefix_length')
        dev_s = grp['deviation_step'].iloc[0]
        detected = grp[grp['p_hijacked'] >= 0.5]
        if not detected.empty:
            det_step = int(detected['prefix_length'].iloc[0])
            detected_flag = True
        else:
            det_step = int(grp['prefix_length'].max()) + 1
            detected_flag = False
        latency = det_step - (dev_s if dev_s is not None else grp['prefix_length'].max()+1)
        pre_action = det_step < (dev_s if dev_s is not None else grp['prefix_length'].max()+1)
        det_records.append({
            'trajectory_id': tid,
            'deviation_step': dev_s,
            'detection_step': det_step,
            'detected': detected_flag,
            'latency': latency,
            'pre_action': pre_action,
        })

    df_det = pd.DataFrame(det_records)
    df_det.to_csv(RESULTS_ROOT / 'latency' / f'transformer_seed{seed}_early_detection.csv', index=False)

    # 11. Save split and experiment config
    with open(RESULTS_ROOT / 'configs' / f'split_seed{seed}.json', 'w') as f:
        json.dump(split, f, indent=2)
    with open(RESULTS_ROOT / 'configs' / f'experiment_config_seed{seed}.json', 'w') as f:
        json.dump({
            'dataset': str(DATA_PATH),
            'seed': seed,
            'embedding_model': EMBEDDING_MODEL_NAME,
            'embedding_dim': 384,
            'transformer': {'model_dim':128,'layers':2,'heads':4,'ff_dim':256,'dropout':0.2},
            'detection_threshold': float(DETECTION_THRESHOLD),
        }, f, indent=2)

    print('\n[COMPLETE] Transformer seed', seed, 'experiment completed. Results written to', RESULTS_ROOT)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force-embed', action='store_true')
    args = p.parse_args()
    run(seed=args.seed, force_embed=args.force_embed)
