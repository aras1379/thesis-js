# rq1/label_comparison_praat_hume.py

import os, json, numpy as np
from scipy.stats import pearsonr, entropy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

LABELS = ['anger','fear','joy','sadness','surprise']

def main():
    praat_labels = []
    hume_labels  = []
    filenames    = []
    praat_vecs   = []
    hume_vecs    = []

    comp_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'comparisons')
    )

    for fn in sorted(os.listdir(comp_dir)):
        data = json.load(open(os.path.join(comp_dir, fn)))
        entry = data.get("entry_id", fn)
        filenames.append(entry)

        # hard labels
        praat_lbl = data.get("praat_label", "unknown")
        hume_lbl  = data.get("hume_label",  "unknown")
        praat_labels.append(praat_lbl)
        hume_labels.append(hume_lbl)

        # soft vectors (normalized)
        h_vec = np.array([ data["hume_probs"].get(e, 0.0)    for e in LABELS ])
        p_vec = np.array([ data["praat_scores"].get(e, 0.0)  for e in LABELS ])
        hume_vecs.append(h_vec)
        praat_vecs.append(p_vec)

    # 0) Per‐file alignment
    df = pd.DataFrame({
        "entry_id":    filenames,
        "praat_label": praat_labels,
        "hume_label":  hume_labels,
        "match":       [p==h for p,h in zip(praat_labels, hume_labels)]
    })
    print("\nPraat vs Hume (hard‑label) comparison:")
    print(df.to_string(index=False))

    # 1) Classification report
    print("\nClassification Report (marker vs Hume):")
    print(classification_report(
        y_true=hume_labels,
        y_pred=praat_labels,
        labels=LABELS,
        zero_division=0
    ))

    # 2) Confusion matrix
    cm = confusion_matrix(hume_labels, praat_labels, labels=LABELS)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Praat label")
    plt.ylabel("Hume label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 3) Cosine similarity over normalized soft‑scores
    cos_sims = []
    for p_vec, h_vec in zip(praat_vecs, hume_vecs):
        denom = np.linalg.norm(p_vec) * np.linalg.norm(h_vec)
        cos_sims.append(np.dot(p_vec, h_vec) / denom if denom>0 else 0.0)
    print(f"\nMean cosine similarity (soft vectors): {np.mean(cos_sims):.3f}")

    # 4) Flattened Pearson r
    all_p = np.vstack(praat_vecs).ravel()
    all_h = np.vstack(hume_vecs).ravel()
    print(f"Flattened Pearson r (soft vectors): {pearsonr(all_p, all_h)[0]:.3f}")

    # 5) Mean JS divergence
    js_ds = []
    for p_vec, h_vec in zip(praat_vecs, hume_vecs):
        m = 0.5*(p_vec + h_vec)
        js = 0.5*(entropy(p_vec, m) + entropy(h_vec, m))
        js_ds.append(js)
    print(f"Mean JS divergence (soft vectors): {np.mean(js_ds):.3f}")

if __name__ == '__main__':
    main()
