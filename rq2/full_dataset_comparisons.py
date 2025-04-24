import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1) Load full RQ2 results
with open("results_combined_rq2_rq3.json", "r") as f:
    data = json.load(f)

EMOS = ["anger", "joy", "sadness", "fear", "surprise"]

# 2) Build clip‑level DataFrame
rows = []
for entry, rec in data.items():
    h = rec.get("hume_emotions", {})
    n = rec.get("nlp_emotions", {})
    row = {"entry_id": entry}
    for emo in EMOS:
        row[f"hume_{emo}"] = h.get(emo, np.nan)
        row[f"nlp_{emo}"]  = n.get(emo, np.nan)
    rows.append(row)

df = pd.DataFrame(rows).set_index("entry_id")

# 3) Compute correlations
corrs = []
for emo in EMOS:
    x = df[f"hume_{emo}"]
    y = df[f"nlp_{emo}"]
    r, p = pearsonr(x, y)
    corrs.append({"emotion": emo, "r": r, "p": p})
corr_df = pd.DataFrame(corrs).set_index("emotion")

# 4) Heatmap of r only
plt.figure(figsize=(6, 4))
sns.heatmap(
    corr_df[["r"]],
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    cbar_kws={"label": "Pearson r"}
)
plt.title("Clip‑level Pearson r: Hume vs NLP, per emotion")
plt.ylabel("Emotion")
plt.tight_layout()
plt.show()

# 5) Print & save separate tables for r and p
os.makedirs("exports", exist_ok=True)

r_df = corr_df[["r"]].rename(columns={"r": "pearson_r"})
p_df = corr_df[["p"]].rename(columns={"p": "p_value"})

print("\n--- Pearson r table ---")
print(r_df.to_string())

print("\n--- p‑value table ---")
print(p_df.to_string())

r_df.to_excel("exports/hume_nlp_correlations_r.xlsx")
p_df.to_excel("exports/hume_nlp_correlations_p.xlsx")
print("\nSaved Excel files:")
print(" • exports/hume_nlp_correlations_r.xlsx")
print(" • exports/hume_nlp_correlations_p.xlsx")

# 6) Scatter + regression per emotion
for emo in EMOS:
    x = df[f"nlp_{emo}"]
    y = df[f"hume_{emo}"]
    plt.figure(figsize=(4, 4))
    sns.regplot(
        x=x, y=y, ci=95,
        scatter_kws={"alpha": 0.6}
    )
    plt.xlabel("NLP Cloud (text‑based)")
    plt.ylabel("Hume AI (speech‑based)")
    plt.title(
        f"{emo.title()}: Hume vs NLP — r={r_df.loc[emo,'pearson_r']:.2f}, "
        f"p={p_df.loc[emo,'p_value']:.3f}"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
