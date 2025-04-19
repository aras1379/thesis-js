import os
import json
from praat_parselmouth.vocal_extract import extract_features

def categorize_emotion_from_vocal_markers(vocal_features):
    pitch = vocal_features["mean_pitch_hz"]
    intensity = vocal_features["mean_intensity_db"]
    hnr = vocal_features["mean_hnr_db"]
    jitter = vocal_features["jitter_local"]
    shimmer = vocal_features["shimmer_local"]
    loudness = intensity
    f1 = vocal_features["formants_hz"]["F1"]

    scores = {
        "anger": 0,
        "fear": 0,
        "joy": 0,
        "sadness": 0,
        "surprise": 0
    }

    # Anger
    if loudness > 65: scores["anger"] += 1
    if shimmer < 0.02: scores["anger"] += 1
    if jitter < 0.02: scores["anger"] += 1

    # Fear
    #if shimmer > 0.03: scores["fear"] += 1
    #if jitter > 0.02: scores["fear"] += 1
    if hnr > 3: scores["fear"] += 1
    if shimmer > 0.03: scores["fear"] += 1
    if jitter > 0.03: scores["fear"] += 1  # increase threshold


    # Joy
    #if pitch > 170: scores["joy"] += 1
    if f1 > 500: scores["joy"] += 1
    if hnr > 3: scores["joy"] += 1
    #if shimmer < 0.02: scores["joy"] += 1
    if pitch > 170: scores["joy"] += 2  # boost joy
    if shimmer < 0.02: scores["joy"] += 1


    # Sadness
    if pitch < 120: scores["sadness"] += 1
    if intensity < 60: scores["sadness"] += 1
    if hnr < 3: scores["sadness"] += 1

    # Surprise
    if loudness < 60: scores["surprise"] += 1
    if shimmer < 0.02: scores["surprise"] += 1
    if jitter < 0.02: scores["surprise"] += 1

    # Return emotion with highest score
    best_label = max(scores.items(), key=lambda x: x[1])[0]

    return best_label

