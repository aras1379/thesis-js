# NOT USE
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define emotions of interest
emotions = ["anger","fear","joy","sadness","surprise"]

# Build comparison DataFrame
records = []
comparisons_dir = "comparisons_rq1"
for fn in sorted(os.listdir(comparisons_dir)):
    if not fn.endswith("_vocal_vs_hume.json"):
        continue
    path = os.path.join(comparisons_dir, fn)
    data = json.load(open(path))
    entry = data.get("entry_id", fn.replace("_vocal_vs_hume.json", ""))
    praat = data.get("praat_scores", {})
    hume  = data.get("hume_probs", {})
    # build row
    row = {"entry_id": entry}
    for emo in emotions:
        row[f"{emo}_praat"] = praat.get(emo, np.nan)
        row[f"{emo}_hume"]  = hume.get(emo, np.nan)
    records.append(row)

df = pd.DataFrame(records).set_index("entry_id")

# Separate into two matrices
praat_df = df[[f"{emo}_praat" for emo in emotions]]
hume_df  = df[[f"{emo}_hume"  for emo in emotions]]

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(df)*0.5)))

for ax, data_mat, title in zip(
    axes, [praat_df.values, hume_df.values],
    ["Praat Scores", "Hume Probabilities"]
):
    im = ax.imshow(data_mat, aspect='auto')
    ax.set_title(title)
    # set ticks
    ax.set_xticks(np.arange(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    # colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()

# Save figure
os.makedirs("exports", exist_ok=True)
fig_path = os.path.join("exports", "praat_hume_comparison_heatmap.png")
fig.savefig(fig_path)
print(f"Saved heatmap: {fig_path}")
