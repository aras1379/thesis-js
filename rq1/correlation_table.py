# correlation_table.py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import load_all_clip_data, compute_feature_arrays
from config import emotions_to_analyze

parser = argparse.ArgumentParser(
    description="Correlation Table for Aggregated Acoustic Features vs. Emotion Scores"
)
parser.add_argument("--emotion", type=str, default="all",
                    help=("Emotion label(s) to analyze (e.g., anger, joy, sadness; "
                          "comma-separated, or 'all' for all emotions)"))
args = parser.parse_args()

# Determine which emotions to proces 
if args.emotion.lower() == "all":
    selected_emotions = emotions_to_analyze
else:
    selected_emotions = [e.strip() for e in args.emotion.split(',')]

# Load data 
data_list = load_all_clip_data("comparisons")
features = compute_feature_arrays(data_list)

# list of acoustic features to correlate.
acoustic_features = [
    ("mean_pitch", "Mean Pitch (Hz)"),
    ("mean_intensity", "Mean Intensity (dB)"),
    ("F1", "F1 (Hz)")
    # Add more features 
]

results = []

for acoustic_key, acoustic_label in acoustic_features:
    for emotion in selected_emotions:
        emotion_key = emotion.lower()
        if emotion_key not in features:
            print(f"Emotion '{emotion}' not present in aggregated features. Skipping...")
            continue

        valid_mask = (~np.isnan(features[acoustic_key])) & (~np.isnan(features[emotion_key]))
        if np.sum(valid_mask) > 1:
            r, p = pearsonr(features[acoustic_key][valid_mask], features[emotion_key][valid_mask])
            results.append({
                "Acoustic Feature": acoustic_label,
                "Emotion": f"Hume {emotion}",
                "Pearson r": round(r, 3),
                "p-value": round(p, 4),
                "Num Points": int(np.sum(valid_mask))
            })
        else:
            results.append({
                "Acoustic Feature": acoustic_label,
                "Emotion": f"Hume {emotion}",
                "Pearson r": np.nan,
                "p-value": np.nan,
                "Num Points": 0
            })

df_results = pd.DataFrame(results, columns=["Acoustic Feature", "Emotion", "Pearson r", "p-value", "Num Points"])
print(df_results.to_string(index=False))
