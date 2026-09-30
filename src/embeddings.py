import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def step_to_text(step):
    text = f"[STATE]\n{step.get('agent_state', '')}\n"
    text += f"[ACTION]\n{step.get('action', '')}\n"
    text += f"[TOOL]\n{step.get('tool_name', '')}\n"
    text += f"[INPUT]\n{step.get('tool_input', '')}\n"
    text += f"[OBSERVATION]\n{step.get('tool_observation', '')}\n"
    text += f"[DECISION]\n{step.get('concise_reasoning_or_decision', '')}"
    return text

def generate_embeddings(trajectories, config):
    model_name = config["embedding_model"]
    emb_dir = config["embeddings_dir"]
    os.makedirs(emb_dir, exist_ok=True)
    cache_path = os.path.join(emb_dir, f"embeddings_{model_name.replace('/', '_')}.npy")
    
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path, allow_pickle=True).item()
        
    print(f"Generating embeddings using {model_name}...")
    model = SentenceTransformer(model_name)
    
    embeddings_dict = {}
    for traj in trajectories:
        traj_id = traj["trajectory_id"]
        texts = [step_to_text(step) for step in traj["steps"]]
        embs = model.encode(texts, convert_to_numpy=True)
        embeddings_dict[traj_id] = embs
        
    np.save(cache_path, embeddings_dict)
    return embeddings_dict