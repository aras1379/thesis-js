#!/usr/bin/env python3
import json, os, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1) Load your RQ2 JSON

# 1) Load JSON
with open("results_combined_rq2_rq3.json", "r") as f:
    data = json.load(f)

EMOS = ["anger", "joy", "sadness", "fear", "surprise"]

# 2) Build DataFrame
rows = []
for entry, rec in data.items():
    h, n = rec["hume_emotions"], rec["nlp_emotions"]
    row = {"entry_id": entry}
    for e in EMOS:
        row[f"hume_{e}"] = h.get(e, np.nan)
        row[f"nlp_{e}"]  = n.get(e, np.nan)
    rows.append(row)

df = pd.DataFrame(rows).set_index("entry_id")

# 3) Compute Pearson r and p-values
corrs = []
for e in EMOS:
    x, y = df[f"hume_{e}"], df[f"nlp_{e}"]
    r, p = pearsonr(x, y)
    corrs.append({"Emotion": e.title(), "Pearson r": round(r, 3), "p-value": round(p, 4)})

corr_df = pd.DataFrame(corrs).set_index("Emotion")

# 4) Print table
print("\n--- Hume vs NLP: Pearson Correlations ---")
print(corr_df.to_string())

# 5) Plot heatmaps (side by side)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

sns.heatmap(corr_df[["Pearson r"]], annot=True, cmap="coolwarm", center=0,
            fmt=".2f", cbar_kws={"label": "Pearson r"}, ax=axes[0])
axes[0].set_title("Hume vs NLP: Pearson r")
axes[0].set_ylabel("Emotion")

sns.heatmap(corr_df[["p-value"]], annot=True, cmap="viridis_r", center=0.05,
            fmt=".3f", cbar_kws={"label": "p-value"}, ax=axes[1])
axes[1].set_title("Hume vs NLP: p-value")
axes[1].set_ylabel("")

plt.tight_layout()

# Save plot
#os.makedirs("exports", exist_ok=True)
plt.savefig("plots_rq2/hume_nlp_correlation_heatmaps.pdf")
plt.show()

# 6) Export BOTH r and p-value to ONE Excel file
excel_path = "exports_rq2/hume_nlp_correlations.xlsx"
corr_df.to_excel(excel_path)
print(f"\nSaved Excel file: {excel_path}")