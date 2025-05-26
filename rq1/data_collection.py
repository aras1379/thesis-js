import os, sys
import json
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_rq1 import INPUT_DIR_OLD, EXPORT_DIR
from config import active_audio_id, audio_files, emotions_to_analyze
from praat_parselmouth.vocal_extract import extract_features
# --- Configuration ---
INPUT_DIR  = INPUT_DIR_OLD
OUTPUT_DIR = EXPORT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# run ONCE in a notebook / REPL
import glob, json, numpy as np
vals = {k: [] for k in
        ("pitch","loud","hnr","jit","shim","F1","F2","F3")}

for entry_id, paths in audio_files.items():
    feats = extract_features(paths["wav"])
    vals["pitch"].append(feats["mean_pitch_st"])
    vals["loud"]. append(feats["mean_intensity_db"])
    vals["hnr"].  append(feats["mean_hnr_db"])
    vals["jit"].  append(feats["jitter_local"])
    vals["shim"]. append(feats["shimmer_local"])
    for F in ("F1","F2","F3"):
        vals[F].append(feats["formants_hz"][F])

REF_STATS = {k: (np.mean(v), np.std(v)) for k,v in vals.items()}
print(json.dumps(REF_STATS, indent=2))

# --- Load and tag records by sentiment ---
records = []
for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".json"):
        continue
    data = json.load(open(os.path.join(INPUT_DIR, fn), encoding="utf-8"))
    eid = data["entry_id"]
    # determine sentiment from entry_id suffix
    sentiment = "negative" if eid.endswith("_neg") else "positive"
    row = {"entry_id": eid, "sentiment": sentiment}
    # 1) Vocal features
    for feat, val in data["vocal_features"].items():
        if feat == "formants_hz":
            for f, v in val.items():
                row[f"formant_{f}_hz"] = v
        else:
            row[feat] = val
    # 2) Praat emotion scores
    for emo, v in data["praat_scores"].items():
        row[f"praat_{emo}"] = v
    # 3) Hume probabilities
    for emo, v in data["hume_probs"].items():
        row[f"hume_{emo}"] = v
    records.append(row)

df = pd.DataFrame(records)

# --- 1) Summary by sentiment ---
for sentiment, group in df.groupby("sentiment"):
    summary = group.drop(columns=["entry_id","sentiment"]).agg(["mean","std"]).T
    summary.columns = ["Mean","Std"]
    summary.to_excel(os.path.join(OUTPUT_DIR, f"dataset_summary_{sentiment}.xlsx"))

# --- 2) Per-clip detail files ---
for sentiment, group in df.groupby("sentiment"):
    clip_dir = os.path.join(OUTPUT_DIR, sentiment)
    os.makedirs(clip_dir, exist_ok=True)
    for eid, sub in group.set_index("entry_id").groupby(level=0):
        detail = sub.drop(columns=["sentiment"]).T.rename(columns={eid: "Value"})
        detail.to_excel(os.path.join(clip_dir, f"{eid}_details.xlsx"))

print(f"Wrote summary and detail files under {OUTPUT_DIR}")
