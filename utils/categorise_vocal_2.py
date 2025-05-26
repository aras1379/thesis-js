import json
import os
from typing import Dict, List, Tuple


    # Anger: loud OR rough above anger’s own norms

# ——— 1) Swedish research raw means & SDs ———
RAW_STATS: Dict[str, Dict[str, float]] = {
    "anger":   {"pitch":5.00, "sd":5.39,  "loud":7.16, "sd_l":0.66, "jit":-0.13, "sd_j":0.38, "shim":-1.03, "sd_s":0.21},
    "fear":    {"pitch":5.81, "sd":2.31,  "loud":5.09, "sd_l":0.71, "jit":-0.98, "sd_j":0.41, "shim":-1.43, "sd_s":0.23},
    "joy":     {"pitch":7.18, "sd":6.25,  "loud":6.49, "sd_l":0.66, "jit":0.58,  "sd_j":0.38, "shim":-1.02, "sd_s":0.21},
    "sadness": {"pitch":3.99, "sd":5.36,  "loud":2.96, "sd_l":0.68, "jit":0.32,  "sd_j":0.39, "shim":-1.02, "sd_s":0.22},
    "surprise":{"pitch":3.56, "sd":4.14,  "loud":1.24, "sd_l":0.68, "jit":2.14,  "sd_j":0.39, "shim":0.13,  "sd_s":0.22},
}

# ——— 2) Build per-emotion thresholds (mean ± 0.5 SD) ———
TH: Dict[str, Dict[str, Dict[str, float]]] = {"pitch": {}, "loud": {}, "jit": {}, "shim": {}}

for emo, st in RAW_STATS.items():
    μp, σp = st["pitch"], st["sd"]
    TH["pitch"][emo] = {"high": μp + 0.5*σp, "low": μp - 0.5*σp}
    μl, σl = st["loud"], st["sd_l"]
    TH["loud"][emo]  = {"high": μl + 0.5*σl, "low": μl - 0.5*σl}
    μj, σj = st["jit"], st["sd_j"]
    TH["jit"][emo]   = {"high": μj + 0.5*σj}
    μs, σs = st["shim"], st["sd_s"]
    TH["shim"][emo]  = {"high": μs + 0.5*σs}


def categorize_emotion_thresholds(vf: Dict[str, float]) -> List[Tuple[str, float]]:
    """
    1) Count threshold‐based cues per emotion.
    2) If no cues fired, fall back to continuous distances.
    3) Otherwise normalize cue counts to scores summing to 1.
    """
    # 1) Unpack
    p = vf.get("mean_pitch_st", 0.0)
    l = vf.get("mean_intensity_db", 0.0)
    j = vf.get("jitter_local", 0.0)
    s = vf.get("shimmer_local", 0.0)

    # 2) Count raw cues
    cues = {emo: 0 for emo in RAW_STATS}

    # Anger: loud or rough
    if l > TH["loud"]["anger"]["high"]: cues["anger"] += 1
    if s > TH["shim"]["anger"]["high"]: cues["anger"] += 1

    # Joy: high pitch & loud
    if p > TH["pitch"]["joy"]["high"]: cues["joy"] += 1
    if l > TH["loud"]["joy"]["high"]:  cues["joy"] += 1

    # Sadness: low pitch & soft
    if p < TH["pitch"]["sadness"]["low"]: cues["sadness"] += 1
    if l < TH["loud"]["sadness"]["low"]:  cues["sadness"] += 1

    # Fear: jittery + mid-high pitch
    if j > TH["jit"]["fear"]["high"]: cues["fear"] += 1
    mid_low = TH["pitch"]["fear"]["low"]
    mid_high= TH["pitch"]["fear"]["high"]
    if mid_low < p <= mid_high:        cues["fear"] += 1

    # Surprise: very high pitch + clean voice
    if p > TH["pitch"]["surprise"]["high"]: cues["surprise"] += 1
    if j < TH["jit"]["surprise"]["high"]*0.5: cues["surprise"] += 1

    # 3) If *all* cues are zero, fall back:
    if sum(cues.values()) == 0:
        fallback = categorize_emotion_table(vf)
        # sort and return fallback directly
        return sorted(fallback.items(), key=lambda kv: kv[1], reverse=True)

    # 4) Otherwise normalize cue counts
    total = sum(cues.values()) or 1
    scores = {emo: round(cnt/total, 3) for emo, cnt in cues.items()}
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

