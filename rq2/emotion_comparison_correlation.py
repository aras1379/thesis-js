import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser(
    description="Correlation Table for Emotion Scores: NLP vs. Hume (Combined File)"
)
parser.add_argument("--emotion", type=str, default="all",
                    help=("Emotion label(s) to analyze (e.g., Anger, Joy, Sadness; "
                          "comma-separated, or 'all' for all emotions)"))
args = parser.parse_args()

all_emotion_categories = ["anger", "joy", "sadness", "fear", "surprise"]
if args.emotion.lower() == "all":
    selected_emotions = all_emotion_categories
else:
    selected_emotions = [e.strip().lower() for e in args.emotion.split(',')]

combined_results_file = "results_combined_normalized_percent.json"
with open(combined_results_file, "r") as f:
    combined_results = json.load(f)

emotion_data = {"hume": {}, "nlp": {}}

for emo in all_emotion_categories:
    emotion_data["hume"][emo] = []
    emotion_data["nlp"][emo] = []

for clip_id, clip_data in combined_results.items():
    hume_raw = clip_data.get("hume_emotions", {})
    hume_emotions = {k.lower(): v for k, v in hume_raw.items() if k.lower() != "time"}
    nlp_raw = clip_data.get("nlp_emotions", {})
    nlp_emotions = {k.lower(): v for k, v in nlp_raw.items()}
    
    for emo in all_emotion_categories:
        emotion_data["hume"][emo].append(hume_emotions.get(emo, np.nan))
        emotion_data["nlp"][emo].append(nlp_emotions.get(emo, np.nan))

# compute the Pearson correlation between Hume and NLP scores
results = []
for emo in selected_emotions:
    hume_array = np.array(emotion_data["hume"][emo])
    nlp_array = np.array(emotion_data["nlp"][emo])
    valid_mask = (~np.isnan(hume_array)) & (~np.isnan(nlp_array))
    if np.sum(valid_mask) > 1:
        r, p = pearsonr(hume_array[valid_mask], nlp_array[valid_mask])
    else:
        r, p = np.nan, np.nan
    results.append({"Emotion": emo.title(), "Pearson r": r, "p-value": p})

df_results = pd.DataFrame(results, columns=["Emotion", "Pearson r", "p-value"])
print(df_results.to_string(index=False))
