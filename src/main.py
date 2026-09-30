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
import numpy as np

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
        report = f"""# Pilot Results Report
    
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
"""
        with open("results/reports/pilot_results.md", "w") as f:
            f.write(report)
            
        print("Pipeline finished successfully!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()