import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load full results
with open("results_combined_rq2_rq3.json","r") as f:
    data = json.load(f)

EMOS = ["anger","joy","sadness","fear","surprise"]

# 1) build DataFrame
rows = []
for entry, rec in data.items():
    h = rec["hume_emotions"]
    n = rec["nlp_emotions"]
    s = rec["self_assessed"]
    row = {"entry_id":entry}
    for emo in EMOS:
        row[f"hume_{emo}"]  = h.get(emo, np.nan)
        row[f"nlp_{emo}"]   = n.get(emo, np.nan)
        row[f"self_{emo}"]  = s.get(emo, np.nan)
    rows.append(row)

df = pd.DataFrame(rows).set_index("entry_id")

# 2) compute correlations
hume_self = []
nlp_self  = []
for emo in EMOS:
    x_h, y_s = df[f"hume_{emo}"], df[f"self_{emo}"]
    r1, p1    = pearsonr(x_h, y_s)
    hume_self.append({"emotion":emo, "r":r1, "p":p1})

    x_n, y_s2 = df[f"nlp_{emo}"], df[f"self_{emo}"]
    r2, p2    = pearsonr(x_n, y_s2)
    nlp_self.append({"emotion":emo, "r":r2, "p":p2})

hs_df = pd.DataFrame(hume_self).set_index("emotion").rename(columns={"r":"r_hs","p":"p_hs"})
ns_df = pd.DataFrame(nlp_self).set_index("emotion").rename(columns={"r":"r_ns","p":"p_ns"})

# 3) heatmaps of r only
os.makedirs("exports", exist_ok=True)

# for each pairing, give the correct column name
for title, corrdf, col in [
    ("Hume vs Self", hs_df, "r_hs"),
    ("NLP  vs Self", ns_df, "r_ns")
]:
    plt.figure(figsize=(5,4))
    sns.heatmap(
        corrdf[[col]],           # pick the right column
        annot=True,
        center=0,
        cmap="coolwarm",
        fmt=".2f",
        cbar_kws={"label": "Pearson r"}
    )
    plt.title(f"Clip‑level r: {title}")
    plt.ylabel("Emotion")
    plt.tight_layout()
    plt.show()

    # save full r/p table for this pairing
    corrdf.to_excel(f"exports/corr_{title.replace(' ','_').lower()}.xlsx")

# 4) scatter + regression per emotion (two panels each)
for emo in EMOS:
    fig, axs = plt.subplots(1,2,figsize=(8,4), sharey=True)
    for ax, (method,label, corrdf) in zip(
      axs,
      [("hume", "Hume vs Self", hs_df),
       ("nlp",  "NLP vs Self", ns_df)]
    ):
        x = df[f"{method}_{emo}"]
        y = df[f"self_{emo}"]
        r = corrdf.loc[emo, f"r_{method[0]}s"]  # r_hs or r_ns
        p = corrdf.loc[emo, f"p_{method[0]}s"]
        sns.regplot(x=x, y=y, ci=95, scatter_kws={"alpha":0.6}, ax=ax)
        ax.set_title(f"{label.title()}\nr={r:.2f}, p={p:.3f}")
        ax.set_xlim(0,1); ax.set_ylim(0,1)  # self_assess scale is ~1–7
        ax.set_xlabel(f"{method.upper()} score"); ax.set_ylabel("Self‑assess")

    plt.suptitle(emo.title())
    plt.tight_layout()
    plt.show()
