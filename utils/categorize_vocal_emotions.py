# utils/categorize_vocal_emotions.py
#File to categorize praat extracts into emotions

#Could be much more improved but i give up now 

from typing import Dict, Tuple, List

FEATURE_STATS = {
    "pitch": {
      "anger": [5.0, 5.39],
      "joy": [7.18, 6.25],
      "fear": [5.81, 2.31],
      "sadness": [3.99, 3.56],
      "surprise": [3.56, 4.14]
    },
    "jitter": {
      "anger": [-0.13, 0.38],
      "joy": [-0.58, 0.38],
      "fear": [-0.98, 0.41],
      "sadness": [0.32, 0.39],
      "surprise": [2.14, 0.39]
    },
    "shimmer": {
      "anger": [-1.03, 0.21],
      "joy": [-1.02, 0.22],
      "fear": [-1.43, 0.23],
      "sadness": [-1.02, 0.22],
      "surprise": [0.13, 0.22]
    },
    "hnr": {
      "anger": [2.36, 0.52],
      "joy": [3.99, 0.52],
      "fear": [4.83, 0.55],
      "sadness": [2.16, 0.54],
      "surprise": [1.31, 0.51]
    },
    "loudness": {
      "anger": [7.16, 0.66],
      "joy": [6.49, 0.66],
      "fear": [5.09, 0.71],
      "sadness": [2.96, 0.68],
      "surprise": [1.24, 0.68]
    },
    "alpha_ratio": {
      "anger": [2.52, 0.4],
      "joy": [2.15, 0.4],
      "fear": [1.14, 0.43],
      "sadness": [1.95, 0.41],
      "surprise": [0.48, 0.41]
    },
    "hammarberg_index": {
      "anger": [-1.57, 0.28],
      "joy": [-1.19, 0.28],
      "fear": [-0.74, 0.28],
      "sadness": [-1.29, 0.27],
      "surprise": [-0.15, 0.27]
    },
    "slopeV0V500": {
      "anger": [2.53, 0.43],
      "joy": [2.68, 0.43],
      "fear": [4.9, 0.46],
      "sadness": [2.76, 0.44],
      "surprise": [1.81, 0.44]
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



def normalize_by_inverse(distances, eps=1e-6):
    inv = {emo: 1.0/(d + eps) for emo, d in distances.items()}
    s = sum(inv.values())
    return {emo: v/s for emo, v in inv.items()}

### STANDARDISED DISTANCE FUNCTION 
def categorize_emotion_table(vocal_features: dict) -> Dict[str, float]:
    """
    1) Computes, for each emotion, the sum of |x - mean|/sd across all features
       defined in FEATURE_STATS (skipping any missing).
    2) Inverts those distances and normalizes so the scores sum to 1.
       
    Returns:
        {emotion: score (0..1), ...}
    """
    # initialize distances to zero
    emotions = next(iter(FEATURE_STATS.values())).keys()
    dists = {emo: 0.0 for emo in emotions}
    
    # mapping from stats‐feature name → key(s) in vocal_features
    feature_map = {
        "pitch":            "mean_pitch_st",
        "jitter":           "jitter_local",
        "shimmer":          "shimmer_local",
        "hnr":              "mean_hnr_db",
        "loudness":         "mean_intensity_db",
        "alpha_ratio":      "alpha_ratio",
        "hammarberg_index": "hammarberg_index",
        "slopeV0V500":      "slopeV0V500",
        "F1":               ("formants_hz", "F1"),
        "F2":               ("formants_hz", "F2"),
        "F3":               ("formants_hz", "F3"),
    }

    for feat_name, emo_stats in FEATURE_STATS.items():
        key = feature_map[feat_name]
        # pull the measured value
        if isinstance(key, tuple):
            val = vocal_features[key[0]].get(key[1])
        else:
            val = vocal_features.get(key)
        if val is None:
            continue

        # *** SCALE FORMANTS DOWN TO kHz ***
        if feat_name in ("F1","F2","F3"):
            val = val / 1000.0

        # now the usual z‐distance
        for emo, (mu, sd) in emo_stats.items():
            if sd == 0:
                continue
            dists[emo] += abs(val - mu) / sd

    # invert & normalize → probabilities
    scores = normalize_by_inverse(dists)
    return scores


FALLBACK_EPSILON = 0.3
MIN_ANGER_ANCHORS = 1
MIN_JOY_ANCHORS = 1
JOY_LOUD_HNR_BONUS = 1.0
K_EXTREME = 1.6 
ANCHOR = 1.0

BENCHMARKS = {
    "anger":    [ ("hnr","below"), ("jit","below"), ("loud", "above")],
    "joy":      [("pitch","above"), ("hnr","above"), ("loud", "above")],
    "sadness":  [("pitch","below"), ("hnr","below"), ("loud", "below")],
    "fear":     [("hnr","above"), ("jit","below"), ("shim","below"), ("pitch","above"),],          # optional
    "surprise": [("jit","above"), ("shim","above")],      
}
FEATURE_WEIGHTS = {
    "pitch": 1.3, "loud": 0.5, "hnr": 1.0,
    "jit":  1.0, "shim": 1.0,
    "f1":   1.0, "f2":   1.0, "f3":  1.0,
}
SD_KEY = {
    "pitch":"sd",  "loud":"sd_l", "hnr":"sd_h",
    "jit":"sd_j",  "shim":"sd_s",
    "f1":"sd_f1",  "f2":"sd_f2",  "f3":"sd_f3",
}

def categorise_emotion_all_scores(
    vf: dict,
    K_NEAR: float = 1.25,
    k_extreme: float = 1.6,
    K_EXTREME_PER_EMO: Dict[str, float] = None,
    use_bm_gate: bool = False, 
) -> List[Tuple[str, float]]:
    # ——————————————————————————————————————
    # vf = extracted features from Praat 
    pitch    = vf.get("mean_pitch_st")
    loud     = vf.get("mean_intensity_db")
    hnr      = vf.get("mean_hnr_db")
    jitter      = vf.get("jitter_local")
    shimmer     = vf.get("shimmer_local")
    formants    = vf.get("formants_hz", {})
    f1, f2, f3  = formants.get("F1"), formants.get("F2"), formants.get("F3")

    # Data from Swedish Research on vocal markers(Ekberg, 2018)
    # Mean and sd
    M = {
    "anger": {
        "pitch": 5.00,   "sd": 5.39,
        "loud": 7.16,    "sd_l": 0.66,
        "hnr": 2.36,     "sd_h": 0.52,
        "jit": -0.13,    "sd_j": 0.38,
        "shim": -1.03,   "sd_s": 0.21,
        "f1": 0.78,      "sd_f1": 0.34,
        "f2": 1.20,      "sd_f2": 0.35,
        "f3": 0.80,      "sd_f3": 0.34
    },
    "fear": {
        "pitch": 5.81,   "sd": 2.31,
        "loud": 5.09,    "sd_l": 0.71,
        "hnr": 4.83,     "sd_h": 0.55,
        "jit": -0.98,    "sd_j": 0.41,
        "shim": -1.43,   "sd_s": 0.23,
        "f1": 1.47,      "sd_f1": 0.37,
        "f2": 1.75,      "sd_f2": 0.37,
        "f3": 0.88,      "sd_f3": 0.37
    },
    "joy": {  
        "pitch": 7.18,   "sd": 6.25,
        "loud": 6.49,    "sd_l": 0.66,
        "hnr": 3.99,     "sd_h": 0.52,
        "jit": 0.58,     "sd_j": 0.38,
        "shim": -1.02,   "sd_s": 0.21,
        "f1": 1.75,      "sd_f1": 0.34,
        "f2": 1.94,      "sd_f2": 0.35,
        "f3": 1.59,      "sd_f3": 0.34
    },
    "sadness": {
        "pitch": 3.99,   "sd": 5.36,
        "loud": 2.96,    "sd_l": 0.68,
        "hnr": 2.16,     "sd_h": 0.54,
        "jit": 0.32,     "sd_j": 0.39,
        "shim": -1.02,   "sd_s": 0.22,
        "f1": 0.12,      "sd_f1": 0.35,
        "f2": 0.23,      "sd_f2": 0.36,
        "f3": -0.10,     "sd_f3": 0.35
    },
    "surprise": {
        "pitch": 3.56,   "sd": 4.14,
        "loud": 1.24,    "sd_l": 0.68,
        "hnr": 1.31,     "sd_h": 0.54,
        "jit": 2.14,     "sd_j": 0.39,
        "shim": 0.13,    "sd_s": 0.22,
        "f1": 0.57,      "sd_f1": 0.35,
        "f2": 1.03,      "sd_f2": 0.36,
        "f3": 0.72,      "sd_f3": 0.35
    }
}
    
    k_ext_per_emo = K_EXTREME_PER_EMO or {}
    default_k = k_extreme
    bm_gate   = use_bm_gate
    
    # Vocal cue helpers
    def near(value, mean, sd, k=K_NEAR):
        return value is not None and abs(value - mean) <= k * sd

    def extreme(value, mean, sd, direction, emo):

        k = k_ext_per_emo.get(emo, default_k)        
        if value is None: return False
        if direction=="above": return value > mean + k*sd
        else:                  return value < mean - k*sd

    def feature_val(key):
        return {
            "pitch": pitch, "loud": loud, "hnr": hnr,
            "jit": jitter, "shim": shimmer,
            "f1": f1 and f1/1000, "f2": f2 and f2/1000, "f3": f3 and f3/1000,
        }[key]

    cue_counts = {emo: 0.0 for emo in M}    
    for emo, stats in M.items():
        hits = 0
        for feat_key, direction in BENCHMARKS[emo]:
            v = feature_val(feat_key)
            if extreme(v, stats[feat_key], stats[SD_KEY[feat_key]], direction, emo):
                hits += 1

        # Benchmark gates 
        if bm_gate and emo in "anger":
            if hits >= 2:         
                cue_counts[emo] += hits * 1.0
        elif bm_gate and emo in ("sadness", "joy"):
            if hits>=1:
                cue_counts[emo] += hits *1.0
        else:
            cue_counts[emo] += hits * 1.0

        for val, key in [
            (pitch, "pitch"), (loud, "loud"), (hnr, "hnr"),
            (jitter, "jit"), (shimmer, "shim"),
            (f1 and f1/1000, "f1"), (f2 and f2/1000, "f2"), (f3 and f3/1000, "f3")
        ]:
            if near(val, stats[key], stats[SD_KEY[key]], K_NEAR):
                cue_counts[emo] += FEATURE_WEIGHTS.get(key, 1.0)
   
    #Call standardised distance function
    prob = categorize_emotion_table(vf)

    combined = {e: cue_counts[e] + 0.3*prob.get(e,0.0) for e in cue_counts}
    return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)

