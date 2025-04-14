import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils import load_all_clip_data, compute_feature_arrays
from config import emotions_to_analyze

parser = argparse.ArgumentParser(
    description="Scatter Plot for Aggregated Acoustic Features vs. Emotion Scores"
)
parser.add_argument("--emotion", type=str, default="all",
                    help=("Emotion label(s) to analyze (e.g., anger, joy, sadness; "
                          "comma-separated, or 'all' for all emotions)"))
args = parser.parse_args()

# Determine the list of emotions to process.
if args.emotion.lower() == "all":
    selected_emotions = emotions_to_analyze
else:
    selected_emotions = [e.strip() for e in args.emotion.split(',')]

data_list = load_all_clip_data("comparisons")
features = compute_feature_arrays(data_list)

available_emotions = [k for k in features.keys() if k not in ("mean_pitch", "mean_intensity", "F1", "F2", "F3")]
print("Available emotion keys:", available_emotions)

def plot_segment_scatter(feature_vals, emotion_vals, feature_label, emotion_label):
    """
    Create a scatter plot for a given acoustic feature versus an emotion.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(feature_vals, emotion_vals, color='purple')
    plt.xlabel(feature_label)
    plt.ylabel(emotion_label)
    plt.title(f"{feature_label} vs {emotion_label}")
    valid = ~np.isnan(feature_vals) & ~np.isnan(emotion_vals)
    if np.sum(valid) > 1:
        m, b = np.polyfit(feature_vals[valid], emotion_vals[valid], 1)
        plt.plot(feature_vals[valid], m * feature_vals[valid] + b, linestyle='--', color='black')
        r, p = pearsonr(feature_vals[valid], emotion_vals[valid])
        plt.legend([f"Pearson r: {r:.2f}, p-value: {p:.3f}"])
    plt.tight_layout()
    plt.show()

for emotion in selected_emotions:
    emotion_key = emotion.lower()

    if emotion_key not in features:
        print(f"Emotion '{emotion}' not present in aggregated features. Available keys: {available_emotions}. Skipping...")
        continue

    # --- Plot for Mean Pitch vs. the selected emotion ---
    plt.figure(figsize=(8,6))
    plt.scatter(features["mean_pitch"], features[emotion_key], color='blue')
    plt.xlabel("Mean Pitch (Hz)")
    plt.ylabel(f"Hume {emotion}")
    plt.title(f"Mean Pitch vs Hume {emotion}")
    if len(features["mean_pitch"]) > 1:
        valid_mask = ~np.isnan(features["mean_pitch"]) & ~np.isnan(features[emotion_key])
        if np.sum(valid_mask) > 1:
            m, b = np.polyfit(features["mean_pitch"][valid_mask], features[emotion_key][valid_mask], 1)
            plt.plot(features["mean_pitch"][valid_mask],
                     m * features["mean_pitch"][valid_mask] + b,
                     linestyle='--', color='black')
            r, p = pearsonr(features["mean_pitch"][valid_mask], features[emotion_key][valid_mask])
            plt.legend([f"Pearson r: {r:.2f}, p-value: {p:.3f}"])
    plt.tight_layout()
    plt.show()
    
    # --- Plot for Mean Intensity vs. the selected emotion ---
    plt.figure(figsize=(8,6))
    plt.scatter(features["mean_intensity"], features[emotion_key], color='orange')
    plt.xlabel("Mean Intensity (dB)")
    plt.ylabel(f"Hume {emotion}")
    plt.title(f"Mean Intensity vs Hume {emotion}")
    if len(features["mean_intensity"]) > 1:
        valid_mask = ~np.isnan(features["mean_intensity"]) & ~np.isnan(features[emotion_key])
        if np.sum(valid_mask) > 1:
            m, b = np.polyfit(features["mean_intensity"][valid_mask], features[emotion_key][valid_mask], 1)
            plt.plot(features["mean_intensity"][valid_mask],
                     m * features["mean_intensity"][valid_mask] + b,
                     linestyle='--', color='black')
            r, p = pearsonr(features["mean_intensity"][valid_mask], features[emotion_key][valid_mask])
            plt.legend([f"Pearson r: {r:.2f}, p-value: {p:.3f}"])
    plt.tight_layout()
    plt.show()
