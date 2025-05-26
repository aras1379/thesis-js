# run_compare_vocal_hume 

import os
import json
from praat_parselmouth.vocal_extract import extract_features
from hume_ai.hume_utils      import load_hume_average, normalize_emotions
from utils.categorize_vocal_emotions import categorise_emotion_all_scores,  categorize_emotion_table
from config import audio_files
from rq1.config_rq1 import INPUT_DIR_V3, EMO_LABELS

OUT_DIR = INPUT_DIR_V3
features_dir = os.path.join(OUT_DIR, "features_cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)


def get_hume_label(probs):
    p = {k.lower(): v for k, v in probs.items() if k.lower() in EMO_LABELS}
    return max(p, key=p.get) if p else "unknown"


def normalize_by_inverse(distances, eps=1e-6):

    inv = {emo: 1.0/(dist + eps) for emo, dist in distances.items()}
    total = sum(inv.values())
    return {emo: score/total for emo, score in inv.items()}


def main():
    for entry_id, paths in audio_files.items():
        wav = paths["wav"]
        feat_file = os.path.join(features_dir, f"{entry_id}_feats.json")

        try:
            # Load or extract features 
            if os.path.exists(feat_file):
                with open(feat_file, 'r') as f:
                    feats = json.load(f)
            else:
                feats = extract_features(wav)
                with open(feat_file, 'w') as f:
                    json.dump(feats, f, indent=4)

            # Rule-based scoring 
            praat_list = categorise_emotion_all_scores(
                feats,
                K_NEAR=1.2,
                k_extreme=1.0,
                K_EXTREME_PER_EMO={
                    "joy":0.7,
                    "anger":1.3,
                    "sadness":1.0,
                    "fear":1.0,
                    "surprise":1.0
                },
      
            )
            praat_scores = {emo: round(score, 2) for emo, score in dict(praat_list).items()}

            # normalize 
            praat_norm = normalize_emotions(praat_scores)
            praat_norm_round = {emo: round(score, 3) for emo, score in praat_norm.items()}

            # pick your top label 
            praat_label = get_hume_label(praat_norm_round)

            # load Hume
            raw_h      = load_hume_average(
                f"hume_ai/filtered_results/average/{entry_id}_average_emotions.json"
            )
            hume_probs = normalize_emotions(raw_h)
            hume_scores = {emo: round(score, 3) for emo, score in hume_probs.items()}
            hume_label = get_hume_label(hume_scores)

            # output combined JSON
            out = {
                "entry_id":       entry_id,
                "vocal_features": feats,
                "praat_scores":   praat_norm_round,
                "praat_label":    praat_label,
                "hume_probs":     hume_scores,
                "hume_label":     hume_label,
            }

            with open(os.path.join(OUT_DIR, f"{entry_id}_vocal_vs_hume.json"), "w") as f:
                json.dump(out, f, indent=4)

            print("ok", entry_id)

        except Exception as e:
            print(f"[!] {entry_id}: {e}")


if __name__ == "__main__":
    main()
