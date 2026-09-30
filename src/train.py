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