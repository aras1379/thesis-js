# rq1/rate_comparison_praat_hume.py
# different comparisons value of normalized praat results and normalized hume results
import os, json, pandas as pd

LABELS = ['anger','fear','joy','sadness','surprise']
rows = []

def normalize(d, keys):
    vals = [d.get(k, 0.0) for k in keys]
    total = sum(vals)
    if total > 0:
        return {k: d.get(k, 0.0)/total for k in keys}
    else:
        return {k: 0.0 for k in keys}
    
for fn in sorted(os.listdir("comparisons_rq1")):
    path = os.path.join("comparisons_rq1", fn)
    data = json.load(open(path))
    entry = data["entry_id"]

    # Hume’s soft‐scores are already normalized by normalize_emotions
    hume_raw = data.get("hume_probs", {})
    hume = normalize(hume_raw, LABELS)

    # Your Praat “soft”‑scores live in the per‑clip JSON as `praat_scores`
    praat_raw = data.get("praat_scores", {})
    praat = normalize(praat_raw, LABELS)

    for emo in LABELS:
        rows.append({
            "entry_id":   entry,
            "emotion":    emo,
            "praat_prob": praat[emo],
            "hume_prob":  hume [emo],
        })

df = pd.DataFrame(rows)

print("\nPraat vs Hume probability distributions (first 10 rows):")
print(df.head(10).to_string(index=False))

wide = df.pivot(index="entry_id", columns="emotion",
                values=["praat_prob","hume_prob"])
print("\nWide‑form example (first 5 clips):")
print(wide.iloc[:5])

print("\nPearson correlations (by emotion):")
for emo in LABELS:
    p = df[df.emotion==emo].praat_prob
    h = df[df.emotion==emo].hume_prob
    r = p.corr(h)
    print(f"  {emo:8s}: r = {r:.3f}")