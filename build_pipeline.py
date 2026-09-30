import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# configs/config.yaml
config_yaml = """
seed: 42
data_path: data/raw/trajectories/traceguard_pilot_v2.jsonl
embeddings_dir: results/embeddings
embedding_model: all-MiniLM-L6-v2
embedding_dim: 384
batch_size: 16
epochs: 30
learning_rate: 0.001
threshold: 0.5
splits:
  train: 0.7
  val: 0.15
  test: 0.15
"""
write_file("configs/config.yaml", config_yaml.strip())

# src/utils.py
utils_py = """
import random
import numpy as np
import torch
import yaml
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data
"""
write_file("src/utils.py", utils_py.strip())

# src/embeddings.py
embeddings_py = """
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def step_to_text(step):
    text = f"[STATE]\\n{step.get('agent_state', '')}\\n"
    text += f"[ACTION]\\n{step.get('action', '')}\\n"
    text += f"[TOOL]\\n{step.get('tool_name', '')}\\n"
    text += f"[INPUT]\\n{step.get('tool_input', '')}\\n"
    text += f"[OBSERVATION]\\n{step.get('tool_observation', '')}\\n"
    text += f"[DECISION]\\n{step.get('concise_reasoning_or_decision', '')}"
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
"""
write_file("src/embeddings.py", embeddings_py.strip())

# src/data_loader.py
data_loader_py = """
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

LABEL_MAP = {"BENIGN": 0, "INJECTION_RESISTED": 1, "HIJACKED": 2}
INV_LABEL_MAP = {0: "BENIGN", 1: "INJECTION_RESISTED", 2: "HIJACKED"}

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, embeddings_dict):
        self.trajectories = trajectories
        self.embeddings_dict = embeddings_dict
        
    def __len__(self):
        return len(self.trajectories)
        
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_id = traj["trajectory_id"]
        emb = self.embeddings_dict[traj_id] # (seq_len, dim)
        label = LABEL_MAP[traj["label"]]
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(label, dtype=torch.long), traj_id

def collate_fn(batch):
    # batch is list of (emb, label, traj_id)
    embs, labels, traj_ids = zip(*batch)
    lengths = [len(e) for e in embs]
    max_len = max(lengths)
    dim = embs[0].shape[1]
    
    padded_embs = torch.zeros(len(batch), max_len, dim)
    for i, e in enumerate(embs):
        padded_embs[i, :len(e), :] = e
        
    return padded_embs, torch.stack(labels), lengths, traj_ids

def create_splits(trajectories, config):
    labels = [t["label"] for t in trajectories]
    
    # Train 70%, Val 15%, Test 15%
    val_test_ratio = config["splits"]["val"] + config["splits"]["test"]
    test_ratio_of_valtest = config["splits"]["test"] / val_test_ratio
    
    train_data, val_test_data, train_labels, val_test_labels = train_test_split(
        trajectories, labels, test_size=val_test_ratio, stratify=labels, random_state=config["seed"]
    )
    
    val_data, test_data, _, _ = train_test_split(
        val_test_data, val_test_labels, test_size=test_ratio_of_valtest, stratify=val_test_labels, random_state=config["seed"]
    )
    
    return train_data, val_data, test_data
"""
write_file("src/data_loader.py", data_loader_py.strip())

# src/models.py
models_py = """
import torch
import torch.nn as nn

class SequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths):
        # x: (batch, seq_len, dim)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.dropout(hn[-1])
        return self.fc(out)

class LightweightTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        # Ensure input_dim is divisible by num_heads. 384 is divisible by 4.
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x, lengths):
        # Create mask for padding
        max_len = x.size(1)
        mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]
        mask = mask.to(x.device)
        
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Mean pooling over valid lengths
        out_pooled = []
        for i, length in enumerate(lengths):
            out_pooled.append(out[i, :length, :].mean(dim=0))
        out_pooled = torch.stack(out_pooled)
        
        return self.fc(out_pooled)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        
import math
"""
write_file("src/models.py", models_py.strip())

# src/train.py
train_py = """
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import copy
import json

def train_model(model, train_loader, val_loader, config, model_name="model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(config["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for embs, labels, lengths, _ in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(embs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * embs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embs, labels, lengths, _ in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                outputs = model(embs, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * embs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"results/checkpoints/{model_name}_best.pth")
    
    with open(f"results/metrics/{model_name}_history.json", "w") as f:
        json.dump(history, f)
        
    return model, history
"""
write_file("src/train.py", train_py.strip())

# src/baseline.py
baseline_py = """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

def extract_features(dataset):
    X = []
    y = []
    traj_ids = []
    for traj in dataset.trajectories:
        traj_id = traj["trajectory_id"]
        embs = dataset.embeddings_dict[traj_id]
        mean_emb = np.mean(embs, axis=0)
        X.append(mean_emb)
        y.append(dataset.LABEL_MAP[traj["label"]])
        traj_ids.append(traj_id)
    return np.array(X), np.array(y), traj_ids

def train_baseline(train_dataset, val_dataset):
    X_train, y_train, _ = extract_features(train_dataset)
    X_val, y_val, _ = extract_features(val_dataset)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    
    os.makedirs("results/checkpoints", exist_ok=True)
    with open("results/checkpoints/baseline_model.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    return clf, val_acc
"""
write_file("src/baseline.py", baseline_py.strip())

# src/evaluate.py
evaluate_py = """
import torch
import numpy as np
import json
import csv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.baseline import extract_features
from src.data_loader import INV_LABEL_MAP

def evaluate_pytorch_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_traj_ids = []
    
    with torch.no_grad():
        for embs, labels, lengths, traj_ids in test_loader:
            embs, labels = embs.to(device), labels.to(device)
            outputs = model(embs, lengths)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_traj_ids.extend(traj_ids)
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_traj_ids

def evaluate_all_models(lstm_model, transformer_model, baseline_model, test_loader, test_dataset):
    results = {}
    
    # LSTM
    y_true_lstm, y_pred_lstm, y_prob_lstm, _ = evaluate_pytorch_model(lstm_model, test_loader)
    results["LSTM"] = compute_metrics(y_true_lstm, y_pred_lstm)
    plot_confusion_matrix(y_true_lstm, y_pred_lstm, "LSTM", "results/figures/cm_lstm.png")
    
    # Transformer
    y_true_trans, y_pred_trans, y_prob_trans, _ = evaluate_pytorch_model(transformer_model, test_loader)
    results["Transformer"] = compute_metrics(y_true_trans, y_pred_trans)
    plot_confusion_matrix(y_true_trans, y_pred_trans, "Transformer", "results/figures/cm_transformer.png")
    
    # Baseline
    X_test, y_true_base, _ = extract_features(test_dataset)
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)
    results["Baseline"] = compute_metrics(y_true_base, y_pred_base)
    plot_confusion_matrix(y_true_base, y_pred_base, "Baseline", "results/figures/cm_baseline.png")
    
    with open("results/metrics/evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        "Accuracy": acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Macro Precision": macro_prec,
        "Macro Recall": macro_rec,
        "Per Class F1": list(f1)
    }

def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BENIGN", "RESISTED", "HIJACKED"], yticklabels=["BENIGN", "RESISTED", "HIJACKED"])
    plt.title(f"{title} Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
"""
write_file("src/evaluate.py", evaluate_py.strip())

# src/early_detection.py
early_detection_py = """
import torch
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import INV_LABEL_MAP

def run_prefix_evaluation(model, trajectories, embeddings_dict, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    prefix_results = []
    
    with torch.no_grad():
        for traj in trajectories:
            traj_id = traj["trajectory_id"]
            true_label = traj["label"]
            dev_step = traj.get("deviation_step", -1)
            inj_step = traj.get("injection_step", -1)
            
            embs = embeddings_dict[traj_id] # (seq_len, dim)
            seq_len = len(embs)
            
            for step_idx in range(1, seq_len + 1):
                prefix_emb = embs[:step_idx]
                prefix_tensor = torch.tensor(prefix_emb, dtype=torch.float32).unsqueeze(0).to(device)
                
                output = model(prefix_tensor, [step_idx])
                probs = torch.softmax(output, dim=1)[0]
                hijack_prob = probs[2].item()
                pred_class = INV_LABEL_MAP[torch.argmax(probs).item()]
                
                prefix_results.append({
                    "Trajectory": traj_id,
                    "Step": step_idx,
                    "True Label": true_label,
                    "Predicted Label": pred_class,
                    "P(Hijacked)": hijack_prob,
                    "Deviation Step": dev_step if dev_step is not None else -1,
                    "Injection Step": inj_step if inj_step is not None else -1
                })
                
    df = pd.DataFrame(prefix_results)
    df.to_csv("results/predictions/prefix_predictions.csv", index=False)
    return df

def analyze_early_detection(df, threshold=0.5):
    hijacked_df = df[df["True Label"] == "HIJACKED"]
    detection_results = []
    
    for traj_id, group in hijacked_df.groupby("Trajectory"):
        group = group.sort_values("Step")
        detected = group[group["P(Hijacked)"] >= threshold]
        
        dev_step = group["Deviation Step"].iloc[0]
        
        if not detected.empty:
            det_step = detected["Step"].iloc[0]
        else:
            det_step = np.nan
            
        latency = det_step - dev_step if pd.notna(det_step) else np.nan
        pre_action = det_step < dev_step if pd.notna(det_step) else False
        
        detection_results.append({
            "Trajectory": traj_id,
            "Deviation Step": dev_step,
            "Detection Step": det_step,
            "Latency": latency,
            "Pre_action": pre_action
        })
        
    det_df = pd.DataFrame(detection_results)
    det_df.to_csv("results/predictions/detection_latency.csv", index=False)
    
    stats = {}
    if not det_df["Latency"].isna().all():
        stats["Mean Latency"] = det_df["Latency"].mean()
        stats["Median Latency"] = det_df["Latency"].median()
        stats["Min Latency"] = det_df["Latency"].min()
        stats["Max Latency"] = det_df["Latency"].max()
    stats["Pre-action Detection Rate"] = det_df["Pre_action"].mean()
    
    # False Positives
    benign_df = df[df["True Label"] == "BENIGN"]
    resisted_df = df[df["True Label"] == "INJECTION_RESISTED"]
    
    benign_fp = benign_df.groupby("Trajectory")["P(Hijacked)"].max() >= threshold
    resisted_fp = resisted_df.groupby("Trajectory")["P(Hijacked)"].max() >= threshold
    
    stats["Benign FPR"] = benign_fp.mean()
    stats["Resisted FPR"] = resisted_fp.mean()
    
    with open("results/metrics/early_detection_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(8,5))
    det_df["Latency"].dropna().hist(bins=10, alpha=0.7)
    plt.title("Detection Latency Distribution")
    plt.xlabel("Latency (Detection Step - Deviation Step)")
    plt.ylabel("Frequency")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig("results/figures/detection_latency.png")
    plt.close()
    
    return stats
"""
write_file("src/early_detection.py", early_detection_py.strip())

# src/main.py
main_py = """
import os
from src.utils import set_seed, load_config, load_jsonl
from src.embeddings import generate_embeddings
from src.data_loader import TrajectoryDataset, DataLoader, collate_fn, create_splits, LABEL_MAP
from src.models import SequenceLSTM, LightweightTransformer
from src.train import train_model
from src.baseline import train_baseline
from src.evaluate import evaluate_all_models
from src.early_detection import run_prefix_evaluation, analyze_early_detection
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import traceback

def plot_history(history, title, path):
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title(f'{title} Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    try:
        config = load_config()
        set_seed(config["seed"])
        
        print("1. Loading dataset...")
        trajectories = load_jsonl(config["data_path"])
        
        # Distributions for report
        class_dist = collections.Counter([t["label"] for t in trajectories])
        len_dist = [len(t["steps"]) for t in trajectories]
        plt.figure()
        plt.bar(class_dist.keys(), class_dist.values())
        plt.title("Class Distribution")
        plt.savefig("results/figures/class_dist.png")
        plt.close()
        
        plt.figure()
        plt.hist(len_dist, bins=range(min(len_dist), max(len_dist)+2), align='left')
        plt.title("Trajectory Length Distribution")
        plt.savefig("results/figures/length_dist.png")
        plt.close()
        
        print("2. Generating Embeddings...")
        embeddings_dict = generate_embeddings(trajectories, config)
        
        print("3. Creating Splits...")
        train_data, val_data, test_data = create_splits(trajectories, config)
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        train_dataset = TrajectoryDataset(train_data, embeddings_dict)
        val_dataset = TrajectoryDataset(val_data, embeddings_dict)
        test_dataset = TrajectoryDataset(test_data, embeddings_dict)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
        
        print("4. Training Baseline...")
        baseline_model, base_acc = train_baseline(train_dataset, val_dataset)
        
        print("5. Training LSTM...")
        lstm = SequenceLSTM(input_dim=config["embedding_dim"], hidden_dim=64)
        lstm, lstm_hist = train_model(lstm, train_loader, val_loader, config, "lstm")
        plot_history(lstm_hist, "LSTM", "results/figures/lstm_history.png")
        
        print("6. Training Transformer...")
        transformer = LightweightTransformer(input_dim=config["embedding_dim"], num_heads=4, hidden_dim=128, num_layers=2)
        transformer, trans_hist = train_model(transformer, train_loader, val_loader, config, "transformer")
        plot_history(trans_hist, "Transformer", "results/figures/transformer_history.png")
        
        print("7. Evaluating Models...")
        eval_results = evaluate_all_models(lstm, transformer, baseline_model, test_loader, test_dataset)
        
        print("8. Early Detection and Localization...")
        # Use LSTM for prefix evaluation as primary sequential model
        prefix_df = run_prefix_evaluation(lstm, test_data, embeddings_dict, config)
        ed_stats = analyze_early_detection(prefix_df, config["threshold"])
        
        # Plot model performance comparison
        models = list(eval_results.keys())
        accs = [eval_results[m]["Accuracy"] for m in models]
        mf1s = [eval_results[m]["Macro F1"] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - width/2, accs, width, label='Accuracy')
        ax.bar(x + width/2, mf1s, width, label='Macro F1')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        plt.title("Model Performance Comparison")
        plt.savefig("results/figures/performance_comparison.png")
        plt.close()
        
        print("9. Generating Report...")
        report = f\"\"\"# Pilot Results Report
    
## 1. Dataset Statistics
Total Trajectories: {len(trajectories)}
Train: {len(train_data)}
Val: {len(val_data)}
Test: {len(test_data)}

## 2. Model Evaluation
{eval_results}

## 3. Early Detection
Mean Latency: {ed_stats.get('Mean Latency', 'N/A')}
Pre-Action Detection Rate: {ed_stats.get('Pre-action Detection Rate', 'N/A')}
Benign False Positive Rate: {ed_stats.get('Benign FPR', 'N/A')}
Resisted False Positive Rate: {ed_stats.get('Resisted FPR', 'N/A')}

*Note: This is a 90-trajectory PILOT dataset. The results are strictly for pilot evaluation.*
\"\"\"
        with open("results/reports/pilot_results.md", "w") as f:
            f.write(report)
            
        print("Pipeline finished successfully!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
write_file("src/main.py", main_py.strip())

print("Files generated.")
