# utils/categorize_vocal_emotions.py
#File to categorize praat extracts into emotions

#Could be much more improved but i give up now 

import os
import json
import sys 
import parselmouth

from praat_parselmouth.vocal_extract import extract_features
from pathlib import Path
import json
import os

import math
from math import log, pi

FEATURE_STATS = {
    "jitter_local": {
        "anger":    (-0.13, 0.38),
        "joy":       (0.58, 0.38),
        "fear":     (-0.98, 0.41),
        "sadness":   (0.32, 0.39),
        "surprise":  (2.14, 0.39),
    },
    "shimmer_local": {
        "anger":  (-1.03, 0.21),
        "joy":    (-1.02, 0.21),
        "fear":   (-1.43, 0.23),
        "sadness":(-1.02, 0.22),
        "surprise":(0.13, 0.22),
    },
    "mean_hnr_db": {
        "anger":   (2.36, 0.52),
        "joy":     (3.99, 0.52),
        "fear":    (4.83, 0.55),
        "sadness": (2.16, 0.54),
        "surprise":(1.31, 0.54),
    },
    "F1": {
        "anger":   (0.78, 0.34),
        "joy":     (1.75, 0.34),
        "fear":    (1.47, 0.37),
        "sadness": (0.12, 0.35),
        "surprise":(0.57, 0.36),
    },
    "F2": {
        "anger":   (1.20, 0.35),
        "joy":     (1.94, 0.35),
        "fear":    (1.75, 0.37),
        "sadness": (0.23, 0.36),
        "surprise":(1.03, 0.36),
    },
    "F3": {
        "anger":   (0.80, 0.34),
        "joy":     (1.59, 0.34),
        "fear":    (0.88, 0.37),
        "sadness": (0.10, 0.35),
        "surprise":(0.72, 0.35),
    },
}

# A single definition of EMOTIONS:
EMOTIONS = ["anger","joy","fear","sadness","surprise"]  # or list(FEATURE_STATS["jitter_local"].keys())

def categorize_emotion_from_table(feats):
    # pull out exactly those six features, converting formants from Hz→kHz
    flat = {
        "jitter_local":   feats["jitter_local"],
        "shimmer_local":  feats["shimmer_local"],
        "mean_hnr_db":    feats["mean_hnr_db"],
        "F1": feats["formants_hz"]["F1"] / 1000.0,
        "F2": feats["formants_hz"]["F2"] / 1000.0,
        "F3": feats["formants_hz"]["F3"] / 1000.0,
    }
    distances = {}
    for emo in EMOTIONS:
        total = 0.0
        for feat, x in flat.items():
            mu, sd = FEATURE_STATS[feat][emo]
            if sd>0 and x is not None and not math.isnan(x):
                z = (x - mu)/sd
                total += z*z
        distances[emo] = total
    return min(distances, key=distances.get)

def classify_segment(audio_path: str, t_mid: float, window: float):
    snd     = parselmouth.Sound(audio_path)
    snippet = snd.extract_part(t_mid-window, t_mid+window, preserve_times=True)
    feats   = extract_features(snippet)

    # build distances exactly as in your table‐classifier
    flat = {
        "jitter_local":   feats["jitter_local"],
        "shimmer_local":  feats["shimmer_local"],
        "mean_hnr_db":    feats["mean_hnr_db"],
        "F1": feats["formants_hz"]["F1"]/1000.0,
        "F2": feats["formants_hz"]["F2"]/1000.0,
        "F3": feats["formants_hz"]["F3"]/1000.0,
    }

    dists = {}
    for emo in EMOTIONS:
        total = 0.0
        for feat, x in flat.items():
            mu, sd = FEATURE_STATS[feat][emo]
            if sd>0 and x is not None and not math.isnan(x):
                z = (x - mu)/sd
                total += z*z
        dists[emo] = total

    print(f"[t={t_mid:.2f}s] distances →", {k:round(v,2) for k,v in dists.items()})
    # pick the minimal distance
    return min(dists, key=dists.get)

def categorize_emotion_from_table_full(sound_or_path, t_mid: float, window: float):
    """
    Returns a dict {emotion: Mahalanobis‑distance} for the snippet
    around t_mid ± window seconds.
    """
    # 1) get a Sound object
    if isinstance(sound_or_path, str):
        snd = parselmouth.Sound(sound_or_path)
    else:
        snd = sound_or_path

    # 2) extract the snippet
    snippet = snd.extract_part(from_time=t_mid - window,
                               to_time  =t_mid + window,
                               preserve_times=True)

    # 3) pull features from that snippet
    feats = extract_features(snippet)

    # flatten out the six table‑features (formants in kHz!)
    flat = {
        "jitter_local":  feats["jitter_local"],
        "shimmer_local": feats["shimmer_local"],
        "mean_hnr_db":   feats["mean_hnr_db"],
        "F1":            feats["formants_hz"]["F1"]  / 1000.0,
        "F2":            feats["formants_hz"]["F2"]  / 1000.0,
        "F3":            feats["formants_hz"]["F3"]  / 1000.0,
    }

    # 4) compute squared‑z Mahalanobis sum for each emotion
    distances = {}
    for emo in EMOTIONS:
        total = 0.0
        for feat_name, x in flat.items():
            mu, sd = FEATURE_STATS[feat_name][emo]
            if sd > 0 and x is not None and not math.isnan(x):
                z = (x - mu) / sd
                total += z*z
        distances[emo] = total

    return distances


def rate_emotion_distances(feats):
    d = {}
    for emo in EMOTIONS:
        s = 0.0
        for feat, stats in FEATURE_STATS.items():
            x = feats.get(feat)
            if x is None or math.isnan(x):
                continue
            mu, sd = stats[emo]
            z = (x - mu)/sd if sd>0 else 0.0
            s += z*z
        d[emo] = s
    return d

def rate_emotion_probs(feats, temp=2.0):
    # naive‑Bayes log‐likelihood
    logps = {}
    for emo in EMOTIONS:
        lp = 0.0
        for feat, stats in FEATURE_STATS.items():
            x = feats.get(feat)
            if x is None or math.isnan(x):
                continue
            mu, sd = stats[emo]
            if sd <= 0: 
                continue
            z = (x - mu)/sd
            lp += -0.5*z*z - math.log(sd*math.sqrt(2*math.pi))
        logps[emo] = lp
    m = max(logps.values())
    exps = {e: math.exp((logps[e]-m)/temp) for e in EMOTIONS}
    S = sum(exps.values()) or 1.0
    return {e: exps[e]/S for e in EMOTIONS}




def categorize_emotion(feats):
    # flatten your formants
    flat = {
        "mean_pitch_st": feats["mean_pitch_st"],
        "mean_intensity_db": feats["mean_intensity_db"],
        "mean_hnr_db": feats["mean_hnr_db"],
        "jitter_local": feats["jitter_local"],
        "shimmer_local": feats["shimmer_local"],
        "F1": feats["formants_hz"]["F1"],
        "F2": feats["formants_hz"]["F2"],
        "F3": feats["formants_hz"]["F3"],
    }

    # compute sum of squared z's for each emotion
    distances = {}
    for emo in EMOTIONS:
        s = 0.0
        for feat, x in flat.items():
            mu, sd = FEATURE_STATS[feat][emo]
            if sd > 0 and x is not None and not math.isnan(x):
                z = (x - mu) / sd
                s += z*z
        distances[emo] = s

    # pick the minimal distance
    return min(distances, key=distances.get)





########Func below NOT in use but might need for documentation later ! #########
def score_emotions_from_markers2(vocal_features):
    # pull out exactly what extract_features returns:
    pitch_st  = vocal_features["mean_pitch_st"]          # semitones above 150Hz
    loudness  = vocal_features["mean_intensity_db"]      # dB
    hnr       = vocal_features["mean_hnr_db"]            # dB
    jitter    = vocal_features["jitter_local"]
    shimmer   = vocal_features["shimmer_local"]

    formants = vocal_features.get("formants_hz", {})
    f1 = formants.get("F1", 0) * 1000
    f2 = formants.get("F2", 0) * 1000
    f3 = formants.get("F3", 0) * 1000

    alpha_ratio      = vocal_features.get("alpha_ratio")
    hammarberg_index = vocal_features.get("hammarberg_index")
    slopeV0V500      = vocal_features.get("slopeV0V500")

    scores = {e:0 for e in ["anger","fear","joy","sadness","surprise"]}

    # ----- pitch rules (no z‑scoring) -----
    if pitch_st >  7.0:  # ≈ 1 semitone above the “joy” mean of 7.18
        scores["joy"] += 1
    if pitch_st <  4.0:  # ≈ 1 semitone below the “sadness” mean of 3.99
        scores["sadness"] += 1

    # ----- anger -----
    if loudness <  5.0 and hnr <  2.5:
        scores["anger"] += 1
    if hammarberg_index is not None and hammarberg_index < -1.0:
        scores["anger"] += 1
    if alpha_ratio is not None and alpha_ratio > 2.3:
        scores["anger"] += 1
    if slopeV0V500 is not None and slopeV0V500 > 2.4:
        scores["anger"] += 1
    if jitter  < 0.02:
        scores["anger"] += 1
    if shimmer < 0.02:
        scores["anger"] += 1

    # ----- fear -----
    if shimmer > 0.03:
        scores["fear"] += 1
    if jitter  > 0.03:
        scores["fear"] += 1
    if hnr     > 4.0:
        scores["fear"] += 1
    if hammarberg_index is not None and hammarberg_index < -0.7:
        scores["fear"] += 1
    if slopeV0V500 is not None and slopeV0V500 > 4.0:
        scores["fear"] += 1
    if alpha_ratio is not None and alpha_ratio < 1.5:
        scores["fear"] += 1

    # ----- joy -----
    if f1  > 500:   scores["joy"] += 1
    if f2  > 1900:  scores["joy"] += 1
    if hnr > 3.0:   scores["joy"] += 1
    if shimmer < 0.02:
        scores["joy"] += 1
    if alpha_ratio  is not None and alpha_ratio > 2.0:
        scores["joy"] += 1
    if hammarberg_index is not None and hammarberg_index > -1.0:
        scores["joy"] += 1
    if slopeV0V500 is not None and slopeV0V500 < 2.5:
        scores["joy"] += 1

    # ----- sadness -----
    if loudness   < 60:    scores["sadness"] += 1
    if hnr        < 3.0:   scores["sadness"] += 1
    if shimmer    > 0.03:  scores["sadness"] += 1
    if alpha_ratio is not None and alpha_ratio < 2.0:
        scores["sadness"] += 1
    if slopeV0V500 is not None and slopeV0V500 < 1.0:
        scores["sadness"] += 1

    # ----- surprise -----
    if hnr      > 3.5:     scores["surprise"] += 1
    if shimmer < 0.02:     scores["surprise"] += 1
    if jitter  < 0.02:     scores["surprise"] += 1
    if slopeV0V500 is not None and slopeV0V500 > 3.0:
        scores["surprise"] += 1
    if hammarberg_index is not None and hammarberg_index > -1.0:
        scores["surprise"] += 1

    return scores



def categorize_emotion_from_vocal_markers(vocal_features):
    pitch = vocal_features["mean_pitch_hz"]
    loudness = vocal_features["mean_intensity_db"]
    hnr = vocal_features["mean_hnr_db"]
    jitter = vocal_features["jitter_local"]
    shimmer = vocal_features["shimmer_local"]
    f1 = vocal_features["formants_hz"]["F1"]
    f2 = vocal_features["formants_hz"]["F2"]
    
    alpha_ratio = vocal_features.get("alpha_ratio")
    hammarberg_index = vocal_features.get("hammarberg_index")
    slopeV0V500 = vocal_features.get("slopeV0V500")
    voiced_length = vocal_features.get("voiced_length")
    unvoiced_length = vocal_features.get("unvoiced_length")

    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    scores = {e: 0 for e in emotions}

    # Selektiva top-3 features per emotion (från tabellen)
    emotion_feature_map = {
        "anger": ["loudness", "hnr", "hammarberg_index"],
        "fear": ["jitter", "shimmer", "hnr"],
        "joy": ["pitch", "f2", "alpha_ratio"],
        "sadness": ["pitch", "loudness", "shimmer"],
        "surprise": ["slopeV0V500", "hammarberg_index", "unvoiced_length"]
    }

    feature_values = {
        "pitch": pitch,
        "loudness": loudness,
        "hnr": hnr,
        "jitter": jitter,
        "shimmer": shimmer,
        "alpha_ratio": alpha_ratio,
        "hammarberg_index": hammarberg_index,
        "slopeV0V500": slopeV0V500,
        "voiced_length": voiced_length,
        "unvoiced_length": unvoiced_length,
        "f2": f2
    }

    for emotion, top_features in emotion_feature_map.items():
        for feat in top_features:
            val = feature_values.get(feat)
            if val is not None and feat in ACOUSTIC_STATS:
                mean, std = ACOUSTIC_STATS[feat][emotion]
                z = z_score(val, mean, std)
                if abs(z) < 1:
                    scores[emotion] += 1

    # Bonusregler för att särskilja joy/sadness bättre
    pitch_z = z_score(pitch, STATS["pitch_mean"], STATS["pitch_std"])
    loudness_z = z_score(loudness, STATS["loudness_mean"], STATS["loudness_std"])

    if pitch_z > 1: scores["joy"] += 1
    if pitch_z < -1: scores["sadness"] += 1
    if loudness_z < -1: scores["sadness"] += 1

    best_label = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Scores: {scores} → {best_label}")
    return best_label


def categorize_emotion_from_vocal_markers4(vocal_features):
    pitch = vocal_features["mean_pitch_hz"]
    loudness = vocal_features["mean_intensity_db"]
    hnr = vocal_features["mean_hnr_db"]
    jitter = vocal_features["jitter_local"]
    shimmer = vocal_features["shimmer_local"]

    f1 = vocal_features["formants_hz"]["F1"]
    f2 = vocal_features["formants_hz"]["F2"]
    f3 = vocal_features["formants_hz"]["F3"]

    alpha_ratio = vocal_features.get("alpha_ratio")
    hammarberg_index = vocal_features.get("hammarberg_index")
    slopeV0V500 = vocal_features.get("slopeV0V500")
    voiced_length = vocal_features.get("voiced_length")
    unvoiced_length = vocal_features.get("unvoiced_length")

    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    scores = {e: 0 for e in emotions}

    # Z-score från din egen data
    pitch_z = z_score(pitch, STATS["pitch_mean"], STATS["pitch_std"])
    loudness_z = z_score(loudness, STATS["loudness_mean"], STATS["loudness_std"])

    # Kombinera ACOUSTIC_STATS för generella drag
    feature_map = {
        "pitch": pitch,
        "loudness": loudness,
        "hnr": hnr,
        "jitter": jitter,
        "shimmer": shimmer,
        "alpha_ratio": alpha_ratio,
        "hammarberg_index": hammarberg_index,
        "slopeV0V500": slopeV0V500,
        "voiced_length": voiced_length,
        "unvoiced_length": unvoiced_length
    }

    for emotion in emotions:
        for feature, value in feature_map.items():
            if value is not None and feature in ACOUSTIC_STATS:
                mean, std = ACOUSTIC_STATS[feature][emotion]
                z = z_score(value, mean, std)
                if abs(z) < 1:
                    scores[emotion] += 1  # mjuk viktning

    # 🔧 Bonusregler med mer spets:
    if pitch_z > 1.0: scores["joy"] += 1
    if pitch_z < -1.0: scores["sadness"] += 1
    if pitch > 170: scores["joy"] += 1
    if pitch < 120: scores["sadness"] += 1

    if f1 > 500: scores["joy"] += 1
    if f2 > 1900: scores["joy"] += 1
    if hnr > 3: scores["joy"] += 1
    if hnr > 4: scores["fear"] += 1
    if shimmer < 0.02: scores["joy"] += 1
    if shimmer > 0.03: scores["fear"] += 1
    if jitter > 0.03: scores["fear"] += 1
    if jitter < 0.02: scores["anger"] += 1
    if alpha_ratio is not None:
        if alpha_ratio > 2.0: scores["joy"] += 1
        if alpha_ratio < 1.5: scores["fear"] += 1
    if hammarberg_index is not None:
        if hammarberg_index > -1.0: scores["joy"] += 1
        if hammarberg_index < -1.0: scores["anger"] += 1
    if slopeV0V500 is not None:
        if slopeV0V500 < 2.5: scores["joy"] += 1
        if slopeV0V500 > 4.0: scores["fear"] += 1
    if unvoiced_length is not None and unvoiced_length < 1.0:
        scores["surprise"] += 1

    best_label = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Scores: {scores} → {best_label}")
    return best_label

def categorize_emotion_from_vocal_markers5(vocal_features):
    # Grundläggande features
    pitch = vocal_features["mean_pitch_hz"]
    loudness = vocal_features["mean_intensity_db"]
    hnr = vocal_features["mean_hnr_db"]
    jitter = vocal_features["jitter_local"]
    shimmer = vocal_features["shimmer_local"]

    f1 = vocal_features["formants_hz"]["F1"]
    f2 = vocal_features["formants_hz"]["F2"]
    f3 = vocal_features["formants_hz"]["F3"]

    # Extra features
    alpha_ratio = vocal_features.get("alpha_ratio")
    hammarberg_index = vocal_features.get("hammarberg_index")
    slopeV0V500 = vocal_features.get("slopeV0V500")
    voiced_length = vocal_features.get("voiced_length")
    unvoiced_length = vocal_features.get("unvoiced_length")

    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    scores = {e: 0 for e in emotions}

    # Alla features vi har i ACOUSTIC_STATS
    feature_map = {
        "pitch": pitch,
        "loudness": loudness,
        "hnr": hnr,
        "jitter": jitter,
        "shimmer": shimmer,
        "alpha_ratio": alpha_ratio,
        "hammarberg_index": hammarberg_index,
        "slopeV0V500": slopeV0V500,
        "voiced_length": voiced_length,
        "unvoiced_length": unvoiced_length
    }

    for emotion in emotions:
        for feature, value in feature_map.items():
            if value is not None and feature in ACOUSTIC_STATS:
                mean, std = ACOUSTIC_STATS[feature][emotion]
                z = z_score(value, mean, std)
                if abs(z) < 1:  # ✅ inom 1 standardavvikelse
                    scores[emotion] += 1

    # Bonusregler som baseras på forskning, kan anpassas vidare:
    if pitch > 170: scores["joy"] += 1
    if pitch < 120: scores["sadness"] += 1
    if f1 > 500: scores["joy"] += 1
    if f2 > 1900: scores["joy"] += 1
    if hnr > 4: scores["fear"] += 1
    if shimmer < 0.02: scores["joy"] += 1
    if shimmer > 0.03: scores["fear"] += 1

    # Resultat
    best_label = max(scores.items(), key=lambda x: x[1])[0]
    print(f"Scores: {scores} → {best_label}")
    return best_label

def categorize_emotion_from_vocal57(vocal_features):
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
    if f1 > 500 and f1 < 700: scores["joy"] += 1  # Add a range check
    if hnr > 3: scores["joy"] += 1
    #if shimmer < 0.02: scores["joy"] += 1
    if pitch > 170: scores["joy"] += 1  # previously += 2


    if shimmer < 0.02: scores["joy"] += 1


    # Sadness
    if pitch < 120: scores["sadness"] += 1
    if intensity < 60: scores["sadness"] += 1
    #if hnr < 3: scores["sadness"] += 1
    if hnr < 5: scores["sadness"] += 1
    if jitter > 0.02: scores["sadness"] += 1


    # Surprise
    if loudness < 60: scores["surprise"] += 1
    if shimmer < 0.02: scores["surprise"] += 1
    if jitter < 0.02: scores["surprise"] += 1

    # Return emotion with highest score
    best_label = max(scores.items(), key=lambda x: x[1])[0]
    print(f"✅ NEW FUNC used | Scores: {scores} → {best_label}")



    return best_label

