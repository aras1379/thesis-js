import os, json
import pandas as pd
import numpy as np

INPUT_DIR = "comparisons_rq1_2"   # folder with your JSONs

# ——— 1) Which IDs are female? (fill in yours) ———
female_ids = {
    "id_004_neg", "id_006_neg", "id_009_neg",
    "id_010_neg", "id_011_neg", "id_013_neg",
    "id_004_pos", "id_006_pos", "id_009_pos",
    "id_010_pos", "id_011_pos", "id_013_pos",
    # …
}

# ——— 2) Load all files into a DataFrame ———
records = []
for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".json"):
        continue
    data = json.load(open(os.path.join(INPUT_DIR, fn), encoding="utf-8"))
    row = {"entry_id": data["entry_id"]}
    # extract only the vocal_features
    vf = data["vocal_features"]
    for feat, val in vf.items():
        if feat == "formants_hz":
            for f, v in val.items():
                row[f"formant_{f}_hz"] = v
        else:
            row[feat] = val
    records.append(row)

df = pd.DataFrame(records)

# ——— 3) Tag gender ———
df["gender"] = df["entry_id"].apply(lambda eid: "female" if eid in female_ids else "male")

# ——— 4) Select only the columns you care about ———
features = [
    "mean_pitch_hz",
    "mean_pitch_st",
    "mean_intensity_db",
    "mean_hnr_db",
    "jitter_local",
    "shimmer_local",
    "formant_F1_hz",
    "formant_F2_hz",
    "formant_F3_hz",
]

# ——— 5) Compute & print summary for each gender ———
for gender, group in df.groupby("gender"):
    summary = group[features].agg([np.mean, np.std]).T
    summary.columns = ["Mean", "Std"]
    summary = summary.round(3)
    print(f"\n=== {gender.upper()} ===")
    print(summary)