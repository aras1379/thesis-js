import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

LABELS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

parser = argparse.ArgumentParser(
    description="Scatter plots of Praat vs. Hume soft scores per emotion"
)
parser.add_argument("--emotion", type=str, default="all",
                    help="Emotion(s) to plot (comma‐separated) or 'all'.")
args = parser.parse_args()

# Determine selected emotions
auto = False
if args.emotion.lower() == 'all':
    selected = LABELS
    auto = True
else:
    selected = [e.strip().lower() for e in args.emotion.split(',')]
    # filter to valid
    selected = [e for e in selected if e in LABELS]

# Load soft scores from comparisons/*.json
comp_dir = os.path.join(os.path.dirname(__file__), '..', 'comparisons_rq1')
records = {emo: {'praat': [], 'hume': []} for emo in LABELS}

for fn in sorted(os.listdir(comp_dir)):
    if not fn.endswith('.json'):
        continue
    data = json.load(open(os.path.join(comp_dir, fn)))
    hume = data.get('hume_probs', {})
    praat = data.get('praat_scores', {})
    # normalize keys
    for emo in LABELS:
        p = praat.get(emo, np.nan)
        h = hume.get(emo, np.nan)
        records[emo]['praat'].append(p)
        records[emo]['hume'].append(h)

# Plot
for emo in selected:
    x = np.array(records[emo]['praat'], dtype=float)
    y = np.array(records[emo]['hume'], dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        print(f"Not enough data for '{emo}' (only {mask.sum()} points), skipping.")
        continue
    xv, yv = x[mask], y[mask]
    # scatter
    plt.figure(figsize=(8,6))
    plt.scatter(xv, yv, alpha=0.7)
    plt.xlabel(f"Praat soft‐score for '{emo}'")
    plt.ylabel(f"Hume soft‐score for '{emo}'")
    plt.title(f"Praat vs. Hume soft scores: {emo}")
    # regression line
    m, b = np.polyfit(xv, yv, 1)
    plt.plot(xv, m*xv + b, linestyle='--', color='black',
             label=f"r={pearsonr(xv,yv)[0]:.2f}, p={pearsonr(xv,yv)[1]:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()