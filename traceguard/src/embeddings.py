"""
embeddings.py - Sentence embedding generation and caching for TRACEGUARD.

Uses sentence-transformers/all-MiniLM-L6-v2.
Each trajectory -> 6 embeddings -> shape [6, D].
Embeddings are cached to avoid recomputation.
"""

import os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_SHORT,
    EMB_CACHE_DIR, STEPS_PER_TRAJECTORY,
)
from src.preprocessing import build_trajectory_texts


_EMBEDDING_DIM = None  # set after first call


def get_embedding_dim() -> int:
    global _EMBEDDING_DIM
    if _EMBEDDING_DIM is None:
        raise RuntimeError("Call generate_embeddings first to set embedding dimension.")
    return _EMBEDDING_DIM


def generate_embeddings(
    trajectories: list[dict],
    model_name: str = EMBEDDING_MODEL_NAME,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """
    Generate or load cached embeddings.

    Returns:
        embeddings_dict: {trajectory_id: np.ndarray of shape [6, D]}
    """
    global _EMBEDDING_DIM

    cache_path = EMB_CACHE_DIR / f"embeddings_{EMBEDDING_MODEL_SHORT}.npy"
    meta_path  = EMB_CACHE_DIR / f"embeddings_{EMBEDDING_MODEL_SHORT}_meta.json"

    if cache_path.exists() and not force_recompute:
        print(f"[Embeddings] Loading cached embeddings from {cache_path}")
        emb_dict = np.load(cache_path, allow_pickle=True).item()
        with open(meta_path) as f:
            meta = json.load(f)
        _EMBEDDING_DIM = meta["embedding_dim"]
        print(f"[Embeddings] Loaded {len(emb_dict)} trajectory embeddings  "
              f"(dim={_EMBEDDING_DIM})")
        return emb_dict

    print(f"[Embeddings] Loading model: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    # Probe dimension
    _EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    print(f"[Embeddings] Embedding dimension: {_EMBEDDING_DIM}")

    emb_dict: dict[str, np.ndarray] = {}
    all_texts = []
    all_ids   = []
    all_idx   = []   # (traj_id, step_position)

    # Collect all step texts (batch encoding for speed)
    print(f"[Embeddings] Building step texts for {len(trajectories)} trajectories ...")
    for traj in trajectories:
        texts = build_trajectory_texts(traj)
        assert len(texts) == STEPS_PER_TRAJECTORY, (
            f"Expected {STEPS_PER_TRAJECTORY} steps, got {len(texts)} "
            f"for {traj['trajectory_id']}"
        )
        for i, txt in enumerate(texts):
            all_texts.append(txt)
            all_ids.append(traj["trajectory_id"])
            all_idx.append(i)

    print(f"[Embeddings] Encoding {len(all_texts)} step texts ...")
    all_embs = model.encode(
        all_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Reassemble per trajectory
    temp: dict[str, list] = {}
    for traj_id, step_i, emb in zip(all_ids, all_idx, all_embs):
        temp.setdefault(traj_id, [None] * STEPS_PER_TRAJECTORY)
        temp[traj_id][step_i] = emb

    for traj_id, emb_list in temp.items():
        emb_dict[traj_id] = np.stack(emb_list, axis=0)  # [6, D]

    # Validate shape
    for traj_id, emb in emb_dict.items():
        assert emb.shape == (STEPS_PER_TRAJECTORY, _EMBEDDING_DIM), (
            f"Bad shape {emb.shape} for {traj_id}"
        )

    # Cache
    np.save(cache_path, emb_dict)
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "embedding_dim": _EMBEDDING_DIM,
            "n_trajectories": len(emb_dict),
            "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        }, f, indent=2)

    print(f"[Embeddings] Cached {len(emb_dict)} embeddings to {cache_path}")
    print(f"[Embeddings] Embedding dimension: {_EMBEDDING_DIM}")
    return emb_dict
