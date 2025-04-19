import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

from scipy.spatial.distance import cosine

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id
from hume_ai.hume_utils import normalize_emotions, combine_surprise_scores
from utils.categorize_vocal_emotions import categorize_emotion_from_vocal_markers

LABEL_LIST = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def one_hot_vector(label, label_list=LABEL_LIST):
    return np.array([1.0 if e == label else 0.0 for e in label_list])

def hume_vector(hume_scores, label_list=LABEL_LIST):
    return np.array([hume_scores.get(e, 0.0) for e in label_list])

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def main():
    parser = argparse.ArgumentParser(description="Compare Praat emotion label to full Hume vector for ONE clip.")
    parser.add_argument("--plot", action="store_true", help="Show bar plot of cosine similarities")
    args = parser.parse_args()

    entry_id = active_audio_id
    comparison_file = f"comparisons/{entry_id}_vocal_vs_hume.json"

    if not os.path.exists(comparison_file):
        print(f"[!] Comparison file not found for {entry_id} at {comparison_file}")
        return

    with open(comparison_file) as f:
        data = json.load(f)

    vocal_features = data["vocal_features"]
    hume_raw = data["hume_emotions"]

    # Combine and normalize
    # Lowercase everything first
    hume_raw = {k.lower(): v for k, v in hume_raw.items()}

# Then combine surprise
    hume_scores = {k.lower(): v for k, v in data["hume_emotions"].items()}
    total = sum(hume_scores.get(k, 0.0) for k in LABEL_LIST)
    assert abs(total - 1.0) < 0.01, "Hume emotion scores are not normalized!"

    praat_label = categorize_emotion_from_vocal_markers(vocal_features)

    results = []
    for label in LABEL_LIST:
        praat_vec = one_hot_vector(label)
        hume_vec = hume_vector(hume_scores)
        sim = cosine_similarity(praat_vec, hume_vec)
        results.append({
            "emotion_label_as_praat": label,
            "praat_is_correct": (label == praat_label),
            "cosine_similarity": round(sim, 4)
        })

    df = pd.DataFrame(results)

    print(f"\n🎧 Active audio ID: {entry_id}")
    print(f"📌 Praat predicted label: {praat_label}")
    print("\n🧮 Cosine similarity between Praat 1-hot and Hume emotion vector:")
    print(df.to_string(index=False))

    if args.plot:
        plt.figure(figsize=(8, 5))
        colors = ['green' if row['praat_is_correct'] else 'gray' for _, row in df.iterrows()]
        plt.bar(df["emotion_label_as_praat"], df["cosine_similarity"], color=colors)
        plt.title(f"Cosine Similarity: Hume vs Praat 1-hot ({entry_id})")
        plt.ylabel("Cosine Similarity")
        plt.xlabel("Emotion label (as if Praat predicted it)")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
