# build_comparisons_with_probs.py

import os
import json
from praat_parselmouth.vocal_extract import extract_features
from hume_ai.hume_utils      import load_hume_average, normalize_emotions
from utils.categorize_vocal_emotions import rate_emotion_distances
from config import audio_files

LABELS = ['anger','fear','joy','sadness','surprise']
out_dir = "comparisons"
os.makedirs(out_dir, exist_ok=True)

def get_hume_label(probs):
    p = {k.lower():v for k,v in probs.items() if k.lower() in LABELS}
    return max(p, key=p.get) if p else "unknown"

def normalize_by_inverse(distances, eps=1e-6):
    """
    Turn a {emo: distance} dict into a normalized {emo: score} 
    by inverting and dividing by the total.
    """
    inv = {emo: 1.0/(dist + eps) for emo, dist in distances.items()}
    total = sum(inv.values())
    return {emo: score/total for emo, score in inv.items()}

def main():
    for entry_id, paths in audio_files.items():
        wav = paths["wav"]
        try:
            feats = extract_features(wav)

            # 1) Mahalanobis–style distances from your table
            praat_distances = rate_emotion_distances(feats)

            # 2) Invert & normalize exactly like Hume does:
            praat_norm = normalize_by_inverse(praat_distances)

            # 3) top Praat label from the normalized inverses
            praat_label = max(praat_norm, key=praat_norm.get)

            # now load Hume
            raw_h      = load_hume_average(
                            f"hume_ai/filtered_results/average/{entry_id}_average_emotions.json")
            hume_probs = normalize_emotions(raw_h)
            hume_label = get_hume_label(hume_probs)

            out = {
                "entry_id":        entry_id,
                "vocal_features":  feats,
                "praat_distances": praat_distances,
                "praat_scores":    praat_norm,
                "praat_label":     praat_label,
                "hume_probs":      hume_probs,
                "hume_label":      hume_label,
            }

            with open(os.path.join(out_dir, f"{entry_id}_vocal_vs_hume.json"), "w") as f:
                json.dump(out, f, indent=4)

            print("✅", entry_id)

        except Exception as e:
            print(f"[!] {entry_id}: {e}")

if __name__ == "__main__":
    main()
