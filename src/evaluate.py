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