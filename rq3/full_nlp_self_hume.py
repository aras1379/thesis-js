#!/usr/bin/env python3
import json, os, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import plot_and_save
# ==== GLOBAL PATHS ====
EXPORT_DIR = "exports_rq3"
PLOTS_DIR = "plots_rq3"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load combined JSON
with open("results_combined_rq2_rq3.json", "r") as f:
    data = json.load(f)

EMOS = ["anger", "joy", "sadness", "fear", "surprise"]

# 1) Build DataFrame
rows = []
for entry, rec in data.items():
    h, n, s = rec["hume_emotions"], rec["nlp_emotions"], rec["self_assessed"]
    row = {"entry_id": entry}
    for e in EMOS:
        row[f"hume_{e}"] = h.get(e, np.nan)
        row[f"nlp_{e}"]  = n.get(e, np.nan)
        row[f"self_{e}"] = s.get(e, np.nan)
    rows.append(row)
df = pd.DataFrame(rows).set_index("entry_id")

# 2) Compute correlations
hs, ns = [], []
for e in EMOS:
    r1, p1 = pearsonr(df[f"hume_{e}"], df[f"self_{e}"])
    r2, p2 = pearsonr(df[f"nlp_{e}"], df[f"self_{e}"])
    hs.append({"emotion": e, "pearson_r": r1, "p_value": p1})
    ns.append({"emotion": e, "pearson_r": r2, "p_value": p2})

hs_df = pd.DataFrame(hs).set_index("emotion")
ns_df = pd.DataFrame(ns).set_index("emotion")

# Print in terminal
print("\n--- Hume vs Self correlations ---")
print(hs_df.to_string())
print("\n--- NLP vs Self correlations ---")
print(ns_df.to_string())

# 3) Heatmaps
for title, df_corr in [
    ("hume_vs_self", hs_df),
    ("nlp_vs_self",  ns_df)
]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    sns.heatmap(df_corr[["pearson_r"]], annot=True, cmap="coolwarm", center=0,
                fmt=".2f", cbar_kws={"label": "Pearson r"}, ax=axes[0])
    axes[0].set_title(f"{title.replace('_',' ').title()}: r")
    axes[0].set_ylabel("Emotion")

    sns.heatmap(df_corr[["p_value"]], annot=True, cmap="viridis_r", center=0.05,
                fmt=".3f", cbar_kws={"label": "p-value"}, ax=axes[1])
    axes[1].set_title(f"{title.replace('_',' ').title()}: p")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plot_and_save(fig, os.path.join(PLOTS_DIR, f"heatmap_{title}"))

    # Export Excel
    df_corr.to_excel(os.path.join(EXPORT_DIR, f"corr_{title}.xlsx"))

# 4) Scatter plots per emotion
for e in EMOS:
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, (method, df_corr) in zip(axs, [("hume", hs_df), ("nlp", ns_df)]):
        x = df[f"{method}_{e}"]
        y = df[f"self_{e}"]
        r = df_corr.loc[e, "pearson_r"]
        p = df_corr.loc[e, "p_value"]
        sns.regplot(x=x, y=y, ci=95, scatter_kws={"alpha": 0.6}, ax=ax)
        ax.set_title(f"{method.upper()} vs Self\nr={r:.2f}, p={p:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"{method.upper()} score")
        if method == "hume":
            ax.set_ylabel("Self-assess")

    plt.suptitle(e.title())
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plot_and_save(fig, os.path.join(PLOTS_DIR, f"scatter_{e}_vs_self"))
