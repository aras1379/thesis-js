import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files
from utils.categorize_vocal_emotions import categorize_emotion_from_vocal_markers
from hume_ai.hume_utils import normalize_emotions, combine_surprise_scores

LABEL_LIST = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def get_hume_top_label(hume_scores: dict) -> str:
    # Lowercase all keys first
    hume_scores = {k.lower(): v for k, v in hume_scores.items()}

    # Keep only label-relevant keys
    emotion_only = {k: v for k, v in hume_scores.items() if k in LABEL_LIST}

    if not emotion_only:
        return "unknown"

    return max(emotion_only.items(), key=lambda item: item[1])[0]



def main():
    praat_labels = []
    hume_labels = []
    ids = []
    skipped_disgust = 0

    for entry_id in audio_files:
        comparison_file = f"comparisons/{entry_id}_vocal_vs_hume.json"
        if not os.path.exists(comparison_file):
            continue

        with open(comparison_file, "r") as f:
            data = json.load(f)

        try:
            praat_label = categorize_emotion_from_vocal_markers(data["vocal_features"])
            hume_label = get_hume_top_label(data["hume_emotions"])

            # ⛔ Skip or count if Hume's top label is 'disgust'
            if hume_label == "disgust":
                skipped_disgust += 1
                continue  # 👈 Optional: skip adding it to evaluation

        except Exception as e:
            print(f"[!] Error processing {entry_id}: {e}")
            continue


        ids.append(entry_id)
        praat_labels.append(praat_label)
        hume_labels.append(hume_label)

    df = pd.DataFrame({
        "entry_id": ids,
        "praat_label": praat_labels,
        "hume_label": hume_labels,
        "match": [p == h for p, h in zip(praat_labels, hume_labels)]
    })
    print(f"\n⚠️ Skipped {skipped_disgust} samples where Hume predicted 'disgust' (not included in vocal features classification)")


    print("\n🎯 Praat vs Hume (hard label) comparison:")
    print(df.to_string(index=False))
    
    

    # Classification report
    print("\n📊 Classification Report (treating Hume as ground truth):")
    print(classification_report(hume_labels, praat_labels, labels=LABEL_LIST, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(hume_labels, praat_labels, labels=LABEL_LIST)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_LIST, yticklabels=LABEL_LIST)
    plt.xlabel("Predicted (Praat)")
    plt.ylabel("True (Hume)")
    plt.title("Confusion Matrix: Praat vs Hume")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
