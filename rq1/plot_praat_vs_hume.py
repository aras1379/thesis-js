import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze
def plot_praat_vs_hume(
    output_dir: str = "comparisons_rq1"
):
    """
    Load a single clip's Praat scores and Hume probabilities,
    plot them side-by-side as a bar chart, and save the figure.

    Parameters:
    - entry_id: the clip identifier
    - results_path: path to the JSON file containing combined results
    - output_dir: directory where the plot image will be saved
    """
    entry_id = active_audio_id
    comp_dir = "comparisons_rq1"
    filename = f"{active_audio_id}_vocal_vs_hume.json"
    fullpath = os.path.join(comp_dir, filename)
    if not os.path.exists(fullpath):
        raise FileNotFoundError(f"No comparison JSON for '{active_audio_id}' at {fullpath}")

# Load the data
    data = json.load(open(fullpath))

    # 2) Extract Praat and Hume data
    praat_scores = data.get("praat_scores", {})
    hume_probs   = {k: v for k, v in data.get("hume_probs", {}).items() if k != "time"}

    # 3) Define the emotions (only those present in both)
    emotions = [emo for emo in praat_scores.keys() if emo in hume_probs]

    # 4) Prepare data vectors
    x = np.arange(len(emotions))
    praat_vals = [praat_scores[emo] for emo in emotions]
    hume_vals  = [hume_probs[emo]   for emo in emotions]

    # 5) Plot
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, praat_vals, width, label="Praat (Audio)")
    ax.bar(x + width/2, hume_vals,  width, label="Hume (Speech AI)")

    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Score")
    ax.set_title(f"Praat vs. Hume Emotion Scores for '{entry_id}'")
    ax.legend()
    plt.tight_layout()

    # 6) Save output image
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{entry_id}_praat_hume_comparison.png")
    fig.savefig(out_path)
    plt.show()

    print(f"Saved comparison plot: {out_path}")

# Example usage:
if __name__ == "__main__":
    # Set your active ID and paths here
    #active_id = "id_005_pos"
    plot_praat_vs_hume()
