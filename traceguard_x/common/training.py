"""Isolated training pipeline for TRACEGUARD-X models.

Strictly follows research integrity rules:
- Early stopping is evaluated ONLY on the validation split (never the test set).
- Best checkpoint is restored and saved to an isolated experiment directory.
- Frozen artifact directories are NEVER written to.
- Preserves full training history and reproducibility metadata.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset

from traceguard_x.common.metrics import LABEL_MAP


class MultiTaskTrajectoryDataset(Dataset):
    """Dataset providing embeddings, main 3-class label, and auxiliary warning label."""

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        embeddings_dict: dict[str, np.ndarray],
        horizon: int = 2,
    ) -> None:
        self.trajectories = trajectories
        self.embeddings_dict = embeddings_dict
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = self.trajectories[idx]
        tid = traj["trajectory_id"]
        emb = self.embeddings_dict[tid]
        label = LABEL_MAP[traj["label"]]

        # Auxiliary target: 1.0 if HIJACKED, 0.0 otherwise.
        # Documented metadata: deviation_step is present for all HIJACKED trajectories.
        is_hijacked = 1.0 if traj["label"] == "HIJACKED" else 0.0

        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(is_hijacked, dtype=torch.float32),
            tid,
        )


def multitask_collate_fn(batch):
    embs, labels, warns, tids = zip(*batch)
    lengths = [e.shape[0] for e in embs]
    max_len = max(lengths)
    dim = embs[0].shape[1]

    padded = torch.zeros(len(batch), max_len, dim, dtype=torch.float32)
    for i, e in enumerate(embs):
        padded[i, :e.shape[0], :] = e

    return padded, torch.stack(labels), torch.stack(warns), lengths, tids


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multitask: bool = False,
    aux_criterion: nn.Module | None = None,
    aux_weight: float = 0.5,
) -> tuple[float, float, float]:
    """Evaluate model on validation loader; returns (val_loss, val_acc, val_macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if is_multitask:
                embs, labels, warns, lengths, _ = batch
                embs, labels, warns = embs.to(device), labels.to(device), warns.to(device)
                logits, warn_logit = model(embs, lengths)
                main_loss = criterion(logits, labels)
                aux_loss = aux_criterion(warn_logit, warns) if aux_criterion else 0.0
                loss = main_loss + aux_weight * aux_loss
            else:
                embs, labels, lengths, _ = batch
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs, lengths)
                loss = criterion(logits, labels)

            total_loss += loss.item() * embs.size(0)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    n = len(val_loader.dataset)
    val_loss = total_loss / n
    val_acc = float(accuracy_score(all_labels, all_preds))
    val_macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return val_loss, val_acc, val_macro_f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_save_dir: Path,
    model_tag: str,
    device: torch.device | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 7,
    is_multitask: bool = False,
    aux_weight: float = 0.5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train model with AdamW and early stopping on validation Macro F1.

    Saves best checkpoint into ``experiment_save_dir / f"{model_tag}_best.pth"``.
    Returns a dict containing:
        'best_epoch': int
        'best_val_f1': float
        'history': dict
        'checkpoint_path': Path
        'training_time_seconds': float
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    main_criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss() if is_multitask else None
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    no_improve = 0
    best_epoch = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    if verbose:
        print(f"\n[Train] Starting {model_tag} on {device}")
        print(f"  LR={learning_rate}  WD={weight_decay}  MaxEpochs={max_epochs}  Patience={patience}")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            if is_multitask:
                embs, labels, warns, lengths, _ = batch
                embs, labels, warns = embs.to(device), labels.to(device), warns.to(device)
                logits, warn_logit = model(embs, lengths)
                loss = main_criterion(logits, labels) + aux_weight * aux_criterion(warn_logit, warns)
            else:
                embs, labels, lengths, _ = batch
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs, lengths)
                loss = main_criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * embs.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss, val_acc, val_f1 = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=main_criterion,
            device=device,
            is_multitask=is_multitask,
            aux_criterion=aux_criterion,
            aux_weight=aux_weight,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        improved = val_f1 > best_val_f1
        marker = " *" if improved else ""
        if verbose:
            print(
                f"  Epoch {epoch:2d}/{max_epochs} | "
                f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}{marker}"
            )

        if improved:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(
                        f"\n[Train] Early stopping at epoch {epoch} "
                        f"(best epoch {best_epoch}, best val F1: {best_val_f1:.4f})"
                    )
                break

    training_time = time.time() - t0

    # Load best weights before returning
    model.load_state_dict(best_weights)
    model.eval()

    # Save best checkpoint into isolated experiment directory
    experiment_save_dir = Path(experiment_save_dir)
    experiment_save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = experiment_save_dir / f"{model_tag}_best.pth"

    checkpoint = {
        "model_config": model.get_config() if hasattr(model, "get_config") else {},
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "training_history": history,
        "training_time_seconds": training_time,
    }
    torch.save(checkpoint, ckpt_path)

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "checkpoint_path": ckpt_path,
        "training_time_seconds": training_time,
    }
