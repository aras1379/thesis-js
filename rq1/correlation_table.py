# correlation_table.py

# takes all emotion labels from praat and hume and prints correlation 
# correlation_table.py

import os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

LABELS = ["anger","fear","joy","sadness","surprise"]

def load_all():
    comp_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__),'..','comparisons_rq1')
    )
    records = []
    for fn in sorted(os.listdir(comp_dir)):
        # **only** process .json files
        if not fn.lower().endswith('.json'):
            continue

        path = os.path.join(comp_dir, fn)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        for emo in LABELS:
            h = data["hume_probs"].get(emo, np.nan)
            p = data["praat_scores"].get(emo, np.nan)
            records.append({
                "clip": fn,
                "emotion": emo,
                "hume_prob":    h,
                "praat_score":  p,
            })
    return pd.DataFrame(records)

def main():
    df = load_all()
    results = []
    for emo in LABELS:
        sub = df[df.emotion == emo].dropna(subset=["hume_prob","praat_score"])
        if len(sub) < 2:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(sub.hume_prob, sub.praat_score)
        results.append({
            "Emotion": emo,
            "Pearson r": round(r,3) if not np.isnan(r) else np.nan,
            "p-value":   round(p,4) if not np.isnan(p) else np.nan,
            "N clips":   len(sub)
        })
    out = pd.DataFrame(results)
    print("\nCorrelation of Hume vs Praat soft‑scores:\n")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
