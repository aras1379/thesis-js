import json
import numpy as np

def load_hume_average(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load Hume average: {e}")
        return {}

def combine_surprise_scores(hume_dict: dict) -> float:
    neg = hume_dict.get("surprise (negative)")
    pos = hume_dict.get("surprise (positive)")
    if neg is None or pos is None or np.isnan(neg) or np.isnan(pos):
        return np.nan
    return (neg + pos) / 2


TARGET_LABELS = {
    "anger", "joy", "sadness", "fear", "surprise"
}

def normalize_emotions(emotion_dict: dict) -> dict:
    emotion_dict = emotion_dict.copy()

    if "surprise (negative)" in emotion_dict and "surprise (positive)" in emotion_dict:
        emotion_dict["surprise"] = combine_surprise_scores(emotion_dict)
        emotion_dict.pop("surprise (negative)", None)
        emotion_dict.pop("surprise (positive)", None)

    # Lowercase everything
    emotion_dict = {k.lower(): v for k, v in emotion_dict.items()}
    
    

    # Keep only target emotions
    filtered = {k: v for k, v in emotion_dict.items() if k in TARGET_LABELS or k == "time"}

    # Normalize
    emotion_scores = {k: v for k, v in filtered.items() if k != "time"}
    total = sum(emotion_scores.values())

    #
    if total > 0:
        normalized = {k: v / total for k, v in emotion_scores.items()}
    else:
        normalized = emotion_scores

    if "time" in filtered:
        normalized["time"] = filtered["time"]

    return normalized
