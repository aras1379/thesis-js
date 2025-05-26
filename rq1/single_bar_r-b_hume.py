## SINGLE CLIP 
## Comparison bar btw rule-based and hume 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze
from utils.data_utils import plot_and_save
from config_rq1 import PLOT_DIR, INPUT_DIR_V3, EMO_LABELS

def plot_praat_vs_hume(
    output_dir = PLOT_DIR
):
    """
    Load a single clip's Praat scores and Hume probabilities,
    plot them side-by-side as a bar chart with fixed emotion order, and save the figure.
    """

    entry_id = active_audio_id
    comp_dir = INPUT_DIR_V3
    filename = f"{entry_id}_vocal_vs_hume.json"
    fullpath = os.path.join(comp_dir, filename)
    if not os.path.exists(fullpath):
        raise FileNotFoundError(f"No comparison JSON for '{entry_id}' at {fullpath}")

    # Load the data
    with open(fullpath) as f:
        data = json.load(f)

    # Extract Praat and Hume data
    praat_scores = data.get("praat_scores", {})
    hume_probs   = {k: v for k, v in data.get("hume_probs", {}).items() if k != "time"}


    # Filter to those present in both, preserving order
    emotions = [e for e in EMO_LABELS if e in praat_scores and e in hume_probs]

    # Prepare data vectors
    x = np.arange(len(emotions))
    praat_vals = [praat_scores[e] for e in emotions]
    hume_vals  = [hume_probs[e]   for e in emotions]

    # Plot
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, praat_vals, width, label="Rule-Based (Vocal Features)")
    ax.bar(x + width/2, hume_vals,  width, label="Hume (Speech AI)")

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Score")
    ax.set_title(f"Rule-Based vs. Hume Emotion Scores for '{entry_id}'")
    ax.legend()
    plt.tight_layout()

    # Save and show via plot_and_save
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{entry_id}_praat_hume_comparison")
    plot_and_save(fig, filepath)

    print(f"Saved comparison plot: {filepath}.pdf")

if __name__ == "__main__":
    plot_praat_vs_hume()
