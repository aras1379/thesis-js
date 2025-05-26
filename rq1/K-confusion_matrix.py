# rq1/label_comparison_praat_hume_by_sentiment.py

import os, sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, entropy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import plot_and_save
from config_rq1 import PLOT_DIR, INPUT_DIR_V3, EMO_LABELS

from common_functions import (SENTIMENTS, group_by_sentiment_files,
                              load_sentiment_records)
LABELS = EMO_LABELS
INPUT_DIR = INPUT_DIR_V3

def analyze_subset(file_list, title):
    """
    Given a list of JSON filenames, load the custom vs. Hume labels and scores,
    and print/plot the comparison results treating Hume as the true labels.
    """
    praat_labels = []
    hume_labels  = []
    praat_vecs   = []
    hume_vecs    = []

    for fn in file_list:
        data = json.load(open(fn, encoding='utf-8'))
        # hard labels
        praat_labels.append(data.get("praat_label","unknown"))
        hume_labels .append(data.get("hume_label", "unknown"))
        # soft vectors
        p_vec = np.array([data["praat_scores"].get(e,0.0) for e in EMO_LABELS])
        h_vec = np.array([data["hume_probs"].get(e,0.0)    for e in EMO_LABELS])
        praat_vecs.append(p_vec)
        hume_vecs.append(h_vec)

    print(f"\n---- {title} ----")
    # per-file hard-label match
    df = pd.DataFrame({
        "Hume (true)":  hume_labels,
        "Praat (pred)": praat_labels,
        "Match?":       [h==p for h,p in zip(hume_labels, praat_labels)]
    })
    print("Hard-label agreement:")
    print(df.value_counts(dropna=False)
          .rename_axis(["Hume (true)","Praat (pred)","Match"])
          .reset_index(name="Count"))

    # classification report
    print("\nClassification report (Hume → Praat):")
    print(classification_report(
        y_true=hume_labels,
        y_pred=praat_labels,
        labels=EMO_LABELS,
        zero_division=0
    ))

    # confusion matrix
    cm = confusion_matrix(
        y_true=hume_labels,
        y_pred=praat_labels,
        labels=EMO_LABELS
    )
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMO_LABELS, yticklabels=EMO_LABELS, ax=ax)
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Praat (predicted)")
    ax.set_ylabel("Hume (true)")
    plt.tight_layout()

    # save it 
    cm_fname = os.path.join(
        PLOT_DIR,
        f"confusion_matrix_{title.replace(' ','_')}"
    )
    plot_and_save(fig, cm_fname)


def main():
    groups = group_by_sentiment_files(INPUT_DIR)
    
    title_map = {
        "all": "All recordings",
        "positive": "Positive recordings",
        "negative": "Negative recordings"
    }
    
    for sentiment in SENTIMENTS:
        files = groups[sentiment]
        print(f"\n {sentiment.upper()} ({len(files)} files)")
        analyze_subset(files, title_map[sentiment])

if __name__=='__main__':
    main()
