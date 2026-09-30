import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from src.data_loader import LABEL_MAP

def extract_features(dataset):
    X = []
    y = []
    traj_ids = []
    for traj in dataset.trajectories:
        traj_id = traj["trajectory_id"]
        embs = dataset.embeddings_dict[traj_id]
        mean_emb = np.mean(embs, axis=0)
        X.append(mean_emb)
        y.append(LABEL_MAP[traj["label"]])
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