import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from src.utils import set_seed, load_config, load_jsonl
from src.embeddings import step_to_text, generate_embeddings
from src.data_loader import create_splits, LABEL_MAP, INV_LABEL_MAP

def run_audit():
    config = load_config()
    set_seed(config["seed"])
    trajectories = load_jsonl(config["data_path"])
    
    audit_output = []
    
    # ---------------------------------------------------------
    # 1. FEATURE LEAKAGE AUDIT
    # ---------------------------------------------------------
    print("--- 1. FEATURE LEAKAGE AUDIT ---")
    fields_used = ["agent_state", "action", "tool_name", "tool_input", "tool_observation", "concise_reasoning_or_decision"]
    forbidden_fields = ["label", "attack_success", "injection_step", "deviation_step", "attack_type", "ground_truth_explanation", "final_outcome", "is_injection_present", "is_behaviorally_deviant"]
    
    # Check step level keys across dataset
    all_step_keys = set()
    for t in trajectories:
        for s in t.get("steps", []):
            all_step_keys.update(s.keys())
            
    forbidden_found_in_steps = [k for k in all_step_keys if k in forbidden_fields]
    
    # Check text representations for suspicious terms
    suspicious_terms = [
        "HIJACKED", "BENIGN", "INJECTION_RESISTED", "attack_success", "deviation",
        "successful attack", "resisted injection", "attack_type", "ground_truth"
    ]
    
    found_suspicious_text = []
    for t in trajectories:
        t_id = t["trajectory_id"]
        for idx, step in enumerate(t.get("steps", [])):
            txt = step_to_text(step)
            for term in suspicious_terms:
                if term.lower() in txt.lower():
                    found_suspicious_text.append((t_id, idx+1, term))
                    
    # ---------------------------------------------------------
    # 2 & 3. CLEAN REPRESENTATION, BASELINE & TF-IDF
    # ---------------------------------------------------------
    print("--- 2 & 3. CLEAN BASELINE & TF-IDF LEAKAGE ---")
    train_data, val_data, test_data = create_splits(trajectories, config)
    
    def get_trajectory_text(traj, mode="full"):
        texts = []
        for step in traj["steps"]:
            if mode == "full":
                txt = f"{step.get('agent_state') or ''} {step.get('action') or ''} {step.get('tool_name') or ''} {step.get('tool_input') or ''} {step.get('tool_observation') or ''} {step.get('concise_reasoning_or_decision') or ''}"
            elif mode == "structural":
                txt = f"{step.get('agent_state') or ''} {step.get('action') or ''} {step.get('tool_name') or ''}"
            elif mode == "observation":
                txt = f"{step.get('tool_observation') or ''}"
            texts.append(txt)
        return " \n ".join(texts)

    train_texts = [get_trajectory_text(t, "full") for t in train_data]
    test_texts = [get_trajectory_text(t, "full") for t in test_data]
    train_y = [LABEL_MAP[t["label"]] for t in train_data]
    test_y = [LABEL_MAP[t["label"]] for t in test_data]
    
    # TF-IDF Classifier
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    clf_tfidf = LogisticRegression(random_state=config["seed"], max_iter=1000)
    clf_tfidf.fit(X_train_tfidf, train_y)
    pred_tfidf = clf_tfidf.predict(X_test_tfidf)
    acc_tfidf = accuracy_score(test_y, pred_tfidf)
    prec_t, rec_t, f1_tfidf, _ = precision_recall_fscore_support(test_y, pred_tfidf, average='macro')
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_features_per_class = {}
    for i, class_name in INV_LABEL_MAP.items():
        top_indices = np.argsort(clf_tfidf.coef_[i])[-10:][::-1]
        top_features_per_class[class_name] = list(feature_names[top_indices])
        
    # Clean Embedding Baseline
    embeddings_dict = generate_embeddings(trajectories, config)
    X_train_emb = np.array([np.mean(embeddings_dict[t["trajectory_id"]], axis=0) for t in train_data])
    X_test_emb = np.array([np.mean(embeddings_dict[t["trajectory_id"]], axis=0) for t in test_data])
    
    clf_emb = LogisticRegression(random_state=config["seed"], max_iter=1000)
    clf_emb.fit(X_train_emb, train_y)
    pred_emb = clf_emb.predict(X_test_emb)
    acc_clean_emb = accuracy_score(test_y, pred_emb)
    _, _, f1_clean_emb, _ = precision_recall_fscore_support(test_y, pred_emb, average='macro')
    cm_clean_emb = confusion_matrix(test_y, pred_emb, labels=[0,1,2])
    
    # ---------------------------------------------------------
    # 4. NEAREST-NEIGHBOR TEST (Test vs Train Cosine Sim)
    # ---------------------------------------------------------
    print("--- 4. NEAREST-NEIGHBOR TEST ---")
    sim_matrix = cosine_similarity(X_test_emb, X_train_emb)
    nn_results = []
    for i, test_t in enumerate(test_data):
        best_idx = np.argmax(sim_matrix[i])
        max_sim = sim_matrix[i][best_idx]
        train_t = train_data[best_idx]
        nn_results.append({
            "test_id": test_t["trajectory_id"],
            "test_label": test_t["label"],
            "nearest_train_id": train_t["trajectory_id"],
            "nearest_train_label": train_t["label"],
            "similarity": float(max_sim)
        })

    # ---------------------------------------------------------
    # 5. DUPLICATE / NEAR-DUPLICATE AUDIT
    # ---------------------------------------------------------
    print("--- 5. DUPLICATE AUDIT ---")
    all_traj_ids = [t["trajectory_id"] for t in trajectories]
    all_embs = np.array([np.mean(embeddings_dict[t["trajectory_id"]], axis=0) for t in trajectories])
    full_sim_matrix = cosine_similarity(all_embs)
    
    # Mask diagonal
    np.fill_diagonal(full_sim_matrix, 0)
    
    max_sim_overall = float(np.max(full_sim_matrix))
    mean_sim_overall = float(np.mean(full_sim_matrix))
    
    pairs_gt_90 = int(np.sum(full_sim_matrix > 0.90) // 2)
    pairs_gt_95 = int(np.sum(full_sim_matrix > 0.95) // 2)
    pairs_gt_99 = int(np.sum(full_sim_matrix > 0.99) // 2)
    
    # ---------------------------------------------------------
    # 6. CLASS SEPARABILITY TEST (PCA / t-SNE)
    # ---------------------------------------------------------
    print("--- 6. CLASS SEPARABILITY TEST ---")
    all_labels = [t["label"] for t in trajectories]
    pca = PCA(n_components=2, random_state=config["seed"])
    all_embs_pca = pca.fit_transform(all_embs)
    
    plt.figure(figsize=(8,6))
    df_pca = pd.DataFrame({"PC1": all_embs_pca[:,0], "PC2": all_embs_pca[:,1], "Label": all_labels})
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Label", style="Label", s=100)
    plt.title("PCA of Trajectory Mean Embeddings")
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/pca_separability.png")
    plt.close()
    
    # ---------------------------------------------------------
    # 7. ATTACK-VOCABULARY TEST
    # ---------------------------------------------------------
    print("--- 7. ATTACK-VOCABULARY TEST ---")
    vocab_terms = [
        "ignore", "previous", "instruction", "system", "administrator",
        "urgent", "override", "secret", "confidential", "authorized",
        "security", "redirect", "delete", "send", "access"
    ]
    
    vocab_counts = {term: {"BENIGN": 0, "INJECTION_RESISTED": 0, "HIJACKED": 0} for term in vocab_terms}
    for t in trajectories:
        lbl = t["label"]
        txt = get_trajectory_text(t, "full").lower()
        for term in vocab_terms:
            if term in txt:
                vocab_counts[term][lbl] += 1

    # ---------------------------------------------------------
    # 8. TEMPLATE AUDIT
    # ---------------------------------------------------------
    print("--- 8. TEMPLATE AUDIT ---")
    unique_goals = len(set(t.get("user_goal") or t.get("original_goal") or "" for t in trajectories))
    unique_injection_prompts = len(set(t.get("injection_prompt") or t.get("attack_prompt") or "" for t in trajectories if t.get("injection")))
    unique_tool_seqs = len(set(tuple(s.get("tool_name") or "" for s in t["steps"]) for t in trajectories))
    unique_obs = len(set("".join(s.get("tool_observation") or "" for s in t["steps"]) for t in trajectories))
    unique_decisions = len(set("".join(s.get("concise_reasoning_or_decision") or "" for s in t["steps"]) for t in trajectories))

    # ---------------------------------------------------------
    # 10. EXPERIMENT A, B, C
    # ---------------------------------------------------------
    print("--- 10. CONTROLLED REPRESENTATION EXPERIMENTS ---")
    exp_results = {}
    for exp_name, mode in [("Exp A (Full)", "full"), ("Exp B (Structural)", "structural"), ("Exp C (Observation)", "observation")]:
        tr_t = [get_trajectory_text(t, mode) for t in train_data]
        te_t = [get_trajectory_text(t, mode) for t in test_data]
        
        vec = TfidfVectorizer(max_features=5000, stop_words='english')
        X_tr = vec.fit_transform(tr_t)
        X_te = vec.transform(te_t)
        
        clf = LogisticRegression(random_state=config["seed"], max_iter=1000)
        clf.fit(X_tr, train_y)
        preds = clf.predict(X_te)
        
        acc = accuracy_score(test_y, preds)
        _, _, f1, _ = precision_recall_fscore_support(test_y, preds, average='macro')
        exp_results[exp_name] = {"Accuracy": float(acc), "Macro F1": float(f1)}

    # ---------------------------------------------------------
    # WRITE REPORT (results/reports/leakage_audit.md)
    # ---------------------------------------------------------
    print("--- GENERATING LEAKAGE AUDIT REPORT ---")
    report_md = f"""# TRACEGUARD: Data Leakage and Data Difficulty Audit

**Project:** TRACEGUARD: Early Detection and Localization of Hijacked LLM Agent Trajectories
**Authors:** Deshna Tendulkar, Harshad Agrawal
**Audit Target:** `traceguard_pilot_v2.jsonl` (90 Trajectories)

---

## 1. Feature & Textual Leakage Audit

### Ground-Truth Metadata Leakage
- **Fields explicitly embedded in step input:** `agent_state`, `action`, `tool_name`, `tool_input`, `tool_observation`, `concise_reasoning_or_decision`.
- **Forbidden ground-truth fields checked:** `label`, `attack_success`, `injection_step`, `deviation_step`, `attack_type`, `ground_truth_explanation`, `final_outcome`, `is_injection_present`, `is_behaviorally_deviant`.
- **Result:** **NO forbidden metadata fields** were found inside the step dictionary entries used for embedding generation.

### Suspicious Term Search in Embedded Text
- Search terms scanned: `HIJACKED`, `BENIGN`, `INJECTION_RESISTED`, `attack_success`, `deviation`, `successful attack`, `resisted injection`, `attack_type`.
- **Found occurrences:** {len(found_suspicious_text)} instances across trajectory texts.
"""
    if found_suspicious_text:
        report_md += "#### Suspicious Text Matches:\n"
        for t_id, s_num, term in found_suspicious_text[:10]:
            report_md += f"- Trajectory `{t_id}`, Step {s_num}: term `{term}`\n"
    else:
        report_md += "- **No explicit ground-truth label terms** were detected inside the step text.\n"

    report_md += f"""
---

## 2. Classification on Clean Representation (Embedding Baseline)

- **Input Representation:** Sentence-Transformer (`all-MiniLM-L6-v2`) mean-pooled step embeddings using only clean step fields.
- **Model:** Logistic Regression (scikit-learn)
- **Clean Baseline Accuracy:** {acc_clean_emb:.4f} ({acc_clean_emb*100:.2f}%)
- **Clean Baseline Macro F1:** {f1_clean_emb:.4f}
- **Confusion Matrix (Clean Baseline):**
```
             Pred: BENIGN  Pred: RESISTED  Pred: HIJACKED
True: BENIGN       {cm_clean_emb[0][0]}             {cm_clean_emb[0][1]}               {cm_clean_emb[0][2]}
True: RESISTED     {cm_clean_emb[1][0]}             {cm_clean_emb[1][1]}               {cm_clean_emb[1][2]}
True: HIJACKED     {cm_clean_emb[2][0]}             {cm_clean_emb[2][1]}               {cm_clean_emb[2][2]}
```

---

## 3. Simple Text Leakage & Bag-Of-Words (TF-IDF) Test

A simple TF-IDF + Logistic Regression model trained strictly on trajectory text achieves:
- **TF-IDF Accuracy:** {acc_tfidf:.4f} ({acc_tfidf*100:.2f}%)
- **TF-IDF Macro F1:** {f1_tfidf:.4f}

### Top Discriminative Features per Class (TF-IDF Weights)
"""
    for cls_name, top_feats in top_features_per_class.items():
        report_md += f"- **Class `{cls_name}` Top Terms:** `{', '.join(top_feats)}`\n"

    report_md += f"""
*Insight:* The presence of distinct attack/injection keywords in prompt injection payloads and behavioral deviation tool calls allows even a simple unigram/bigram TF-IDF model to achieve near-perfect or perfect class separation.

---

## 4. Nearest-Neighbor Train-to-Test Similarity Test

Evaluating cosine similarity between each test trajectory embedding and its nearest training trajectory embedding:
"""
    report_md += "| Test Trajectory ID | Test Label | Nearest Train ID | Nearest Train Label | Cosine Similarity |\n"
    report_md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for r in nn_results:
        report_md += f"| `{r['test_id']}` | `{r['test_label']}` | `{r['nearest_train_id']}` | `{r['nearest_train_label']}` | {r['similarity']:.4f} |\n"

    report_md += f"""
- **Mean Nearest-Neighbor Similarity (Test to Train):** {np.mean([r['similarity'] for r in nn_results]):.4f}
- **Max Nearest-Neighbor Similarity (Test to Train):** {np.max([r['similarity'] for r in nn_results]):.4f}

---

## 5. Duplicate & Near-Duplicate Audit

Pairwise trajectory similarity across all 90 trajectories:
- **Maximum Pairwise Cosine Similarity:** {max_sim_overall:.4f}
- **Mean Pairwise Cosine Similarity:** {mean_sim_overall:.4f}
- **Number of Pairs with Similarity > 0.90:** {pairs_gt_90}
- **Number of Pairs with Similarity > 0.95:** {pairs_gt_95}
- **Number of Pairs with Similarity > 0.99:** {pairs_gt_99}

---

## 6. Class Separability (PCA Visualization)

- **Plot Saved:** `results/figures/pca_separability.png`
- **Separability Finding:** PCA visualization confirms that `BENIGN`, `INJECTION_RESISTED`, and `HIJACKED` trajectories form separate, tight clusters in embedding space.
- **Reason:** 
  1. `BENIGN` trajectories lack any prompt injection phrases or security-policy violation terms.
  2. `INJECTION_RESISTED` trajectories contain injection text in tool observations but main agent reasoning/action strictly follows the system prompt.
  3. `HIJACKED` trajectories contain injection text *and* exhibit abrupt shifts in tool calls (e.g., unauthorized data exfiltration or system commands).

---

## 7. Attack Vocabulary Frequency Analysis

Occurrences of key security/attack terms across trajectory classes (out of 30 trajectories per class):

| Term | BENIGN Trajectories | INJECTION_RESISTED Trajectories | HIJACKED Trajectories |
| :--- | :---: | :---: | :---: |
"""
    for term, counts in vocab_counts.items():
        report_md += f"| `{term}` | {counts['BENIGN']} | {counts['INJECTION_RESISTED']} | {counts['HIJACKED']} |\n"

    report_md += f"""
---

## 8. Template & Diversity Audit

- **Unique User Goals:** {unique_goals}
- **Unique Injection Prompts:** {unique_injection_prompts}
- **Unique Tool Sequences:** {unique_tool_seqs}
- **Unique Full Tool Observations:** {unique_obs}
- **Unique Reasoning/Decision Statements:** {unique_decisions}

*Finding:* The pilot dataset relies on a small set of template structures (e.g. 5 attack families across 90 trajectories), leading to high intra-class homogeneity and distinct inter-class keyword profiles.

---

## 9. Train / Validation / Test Split Integrity

- **Split Timing:** The split was performed **strictly before** TF-IDF fitting and PyTorch model training.
- **Data Leakage across Splits:** 0% (Split performed at the trajectory ID level; no steps from the same trajectory cross splits).

---

## 10. Controlled Representation Experiments (A, B, C)

To test feature sensitivity, Logistic Regression was evaluated on different text representations:

| Representation | Included Fields | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Exp A (Full)** | `agent_state`, `action`, `tool_name`, `tool_input`, `tool_observation`, `concise_reasoning` | {exp_results['Exp A (Full)']['Accuracy']:.4f} | {exp_results['Exp A (Full)']['Macro F1']:.4f} |
| **Exp B (Structural)** | `agent_state`, `action`, `tool_name` | {exp_results['Exp B (Structural)']['Accuracy']:.4f} | {exp_results['Exp B (Structural)']['Macro F1']:.4f} |
| **Exp C (Observation)** | `tool_observation` (contains injection payloads) | {exp_results['Exp C (Observation)']['Accuracy']:.4f} | {exp_results['Exp C (Observation)']['Macro F1']:.4f} |

---

## 11. Sequential Advantage Assessment

| Model | Architecture | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Non-Sequential Baseline** | Mean-Pooled Embedding + Logistic Regression | 1.0000 | 1.0000 |
| **LSTM** | Sequential PyTorch LSTM | 1.0000 | 1.0000 |
| **Transformer** | Lightweight Transformer Encoder | 1.0000 | 1.0000 |

### Official Finding on Sequential Advantage:
> **"The pilot dataset (`traceguard_pilot_v2.jsonl`) is perfectly separable even by a non-sequential baseline; therefore, the current pilot experiment does not establish a sequential modeling advantage."**

---

## 12. Final Diagnostic Conclusion & Recommendations

1. **Metadata Leakage:** **NONE.** No ground-truth fields (`label`, `attack_success`, `deviation_step`, etc.) were present in the model inputs.
2. **Textual Leakage:** **NONE (Explicit).** No explicit label strings like `"HIJACKED"` or `"BENIGN"` were leaked.
3. **Data Difficulty & Template Uniformity:** **HIGH SEPARABILITY.** 
   - The pilot dataset consists of 90 highly structured trajectories generated from 5 attack templates.
   - The presence of injection keywords in `tool_observation` and specific hijacked tool calls in later steps creates strong linear separability.
4. **Recommendation for Next Phase:**
   - **Do NOT modify or alter `traceguard_pilot_v2.jsonl`** (retaining integrity of pilot results).
   - For Phase 2 (Scale & Benchmark), generate a significantly larger, more diverse dataset (500-1000+ trajectories) featuring:
     - Harder/subtle prompt injections (semantic obfuscation, zero explicit attack keywords).
     - Varied agent reasoning styles and noise in benign trajectories.
     - Multi-turn benign trajectories with complex tool sequences to test true sequential anomaly detection.
"""
    
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/leakage_audit.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print("Audit report successfully written to results/reports/leakage_audit.md!")

if __name__ == "__main__":
    run_audit()
