"""
train_lstm.py - Training loop for TraceGuardLSTM.

Early stopping based on validation Macro F1 (not test).
Best checkpoint is saved and restored.
"""

import copy, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

from src.config import (
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE, MODELS_DIR,
)
from src.utils import get_device


def train_lstm(
    model,
    train_loader,
    val_loader,
    device=None,
    learning_rate: float = LEARNING_RATE,
    weight_decay:  float = WEIGHT_DECAY,
    max_epochs:    int   = MAX_EPOCHS,
    patience:      int   = EARLY_STOPPING_PATIENCE,
    model_tag:     str   = "lstm_seed42",
) -> tuple:
    """
    Train model with AdamW + CrossEntropyLoss.
    Early stopping on validation Macro F1.

    Returns:
        model (best checkpoint loaded)
        history dict
    """
    if device is None:
        device = get_device()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_f1   = -1.0
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0
    best_epoch    = 0

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [],
    }

    print(f"\n[Train] Starting training: {model_tag}")
    print(f"  LR={learning_rate}  WD={weight_decay}  Batch={BATCH_SIZE}  "
          f"MaxEpochs={max_epochs}  Patience={patience}")
    print(f"  Device: {device}\n")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        # ── Train ──────────────────────────────
        model.train()
        train_loss = 0.0
        for embs, labels, lengths, _ in train_loader:
            embs   = embs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(embs, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * embs.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate ────────────────────────────
        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        improved = val_f1 > best_val_f1
        marker   = " *" if improved else ""
        print(f"  Epoch {epoch:3d}/{max_epochs} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
              f"ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}{marker}")

        if improved:
            best_val_f1  = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
            best_epoch   = epoch
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Train] Early stopping at epoch {epoch} "
                      f"(best epoch={best_epoch}, best val F1={best_val_f1:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\n[Train] Training complete in {elapsed:.1f}s | "
          f"Best epoch: {best_epoch} | Best val Macro F1: {best_val_f1:.4f}")

    # Restore best weights
    model.load_state_dict(best_weights)

    # Save checkpoint
    ckpt_path = MODELS_DIR / f"{model_tag}_best.pth"
    torch.save({
        "model_state_dict": best_weights,
        "best_epoch":       best_epoch,
        "best_val_f1":      best_val_f1,
        "model_config":     model.get_config(),
    }, ckpt_path)
    print(f"[Train] Checkpoint saved: {ckpt_path}")

    history["best_epoch"]   = best_epoch
    history["best_val_f1"]  = best_val_f1
    history["training_seconds"] = elapsed

    return model, history


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for embs, labels, lengths, _ in loader:
            embs   = embs.to(device)
            labels = labels.to(device)
            logits = model(embs, lengths)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * embs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1
