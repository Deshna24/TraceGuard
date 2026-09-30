"""
utils.py - Shared utility helpers for TRACEGUARD.
"""

import os, random, json
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN (minor speed cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
        print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_dict(d: dict, title: str = ""):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:45s} {v:.4f}")
        else:
            print(f"  {k:45s} {v}")
    print()
