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
    neg = hume_dict.get("Surprise (negative)")
    pos = hume_dict.get("Surprise (positive)")
    if neg is None or pos is None or np.isnan(neg) or np.isnan(pos):
        return np.nan
    return (neg + pos) / 2

def normalize_emotions(emotion_dict: dict) -> dict:
    """
    Normalize emotion scores to sum to 1 and round to 3 decimals.
    Combines surprise (positive/negative) into a single 'surprise' key.
    Keeps 'time' unchanged.
    """
    emotion_dict = emotion_dict.copy()

    if "Surprise (negative)" in emotion_dict and "Surprise (positive)" in emotion_dict:
        emotion_dict["surprise"] = combine_surprise_scores(emotion_dict)
        emotion_dict.pop("Surprise (negative)", None)
        emotion_dict.pop("Surprise (positive)", None)

    # Lowercase 
    normalized_dict = {
        (k.lower() if k.lower() != "time" else "time"): v
        for k, v in emotion_dict.items()
    }

    emotion_scores = {k: v for k, v in normalized_dict.items() if k != "time"}
    total = sum(emotion_scores.values())

    if total > 0:
        normalized_scores = {
            k: round(v / total, 3) for k, v in emotion_scores.items()
        }
    else:
        normalized_scores = emotion_scores

    if "time" in normalized_dict:
        normalized_scores["time"] = normalized_dict["time"]

    return normalized_scores
