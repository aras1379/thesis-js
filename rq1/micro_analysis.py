import os, json, math, argparse, sys
import numpy as np, pandas as pd, parselmouth
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze
from utils.categorize_vocal_emotions import classify_segment, categorize_emotion_from_table_full

LABEL_LIST = ['anger','fear','joy','sadness','surprise']
WINDOW     = 0.5  # +/- 0.5s snippet

def segment_average(x, v, t_mid, window=WINDOW):
    mask = (x >= t_mid - window) & (x <= t_mid + window)
    return np.nanmean(v[mask]) if np.any(mask) else np.nan

def get_pitch_data(wav):
    snd = parselmouth.Sound(wav)
    p   = snd.to_pitch()
    t   = p.xs()
    v   = p.selected_array['frequency']
    v[v==0] = np.nan
    return t, v

def get_intensity_data(wav):
    snd = parselmouth.Sound(wav)
    I   = snd.to_intensity()
    t   = I.xs()
    v   = I.values.T.squeeze()
    v[v==0] = np.nan
    return t, v

def load_hume_segments(path):
    return json.load(open(path))


def normalize_dict(d, keys):
    """Take d[key] over keys, and return a new dict of key→(d[key]/sum)."""
    vals = np.array([ d.get(k,0.0) for k in keys ], dtype=float)
    total = vals.sum()
    if total > 0:
        probs = vals/total
    else:
        probs = np.zeros_like(vals)
    return dict(zip(keys, probs))

def micro_level_analysis_all2(wav, hume_json_path):
    pt, pv = get_pitch_data(wav)
    it, iv = get_intensity_data(wav)
    segments = load_hume_segments(hume_json_path)

    rows = []
    for seg in segments:
        t_mid = seg.get("time")
        if t_mid is None: 
            continue

        row = {
            "SegmentTime":    t_mid,
            "AvgPitch_Hz":    segment_average(pt, pv, t_mid),
            "AvgIntensity_dB":segment_average(it, iv, t_mid),
        }

        # 1) dump all Hume scores
        for emo in LABEL_LIST:
            # keys in seg might be capitalized or have other fields—lowercase match
            row[f"hume_{emo}"] = seg.get(emo, seg.get(emo.capitalize(), np.nan))

        # 2) pick Hume top‐label
        hvec = {emo: row[f"hume_{emo}"] for emo in LABEL_LIST}
        row["Hume_label"] = max(hvec, key=lambda e: hvec[e] or 0)

        # 3) Praat label on the same snippet
        row["Praat_label"] = classify_segment(wav, t_mid, WINDOW)

        rows.append(row)

    return pd.DataFrame(rows)

def micro_level_analysis_all(wav, hume_json_path):
    """
    Returns a DataFrame per Hume segment with:
      - SegmentTime
      - AvgPitch_Hz, AvgIntensity_dB
      - hume_prob_<emo> + Hume_label
      - praat_prob_<emo> + Praat_label
    """
    pt, pv = get_pitch_data(wav)
    it, iv = get_intensity_data(wav)
    segments = load_hume_segments(hume_json_path)

    rows = []
    for seg in segments:
        t_mid = seg.get("time")
        if t_mid is None:
            continue

        row = {
            "SegmentTime":     t_mid,
            "AvgPitch_Hz":     segment_average(pt, pv, t_mid),
            "AvgIntensity_dB": segment_average(it, iv, t_mid),
        }

        # — Hume soft‑scores (normalized) —
        raw_hume = {
            emo: seg.get(emo, seg.get(emo.capitalize(), 0.0))
            for emo in LABEL_LIST
        }
        hume_probs = normalize_dict(raw_hume, LABEL_LIST)
        for emo, p in hume_probs.items():
            row[f"hume_prob_{emo}"] = p
        row["Hume_label"] = max(hume_probs, key=hume_probs.get)

        # — Praat soft‑scores (invert Mahalanobis, normalize) —
        dists = categorize_emotion_from_table_full(wav, t_mid, WINDOW)
        inv   = { emo: 1.0/(d+1e-8) for emo,d in dists.items() }
        praat_probs = normalize_dict(inv, LABEL_LIST)
        for emo, p in praat_probs.items():
            row[f"praat_prob_{emo}"] = p
        row["Praat_label"] = max(praat_probs, key=praat_probs.get)

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Segment‑Level Table (soft‑scores only)"
    )
    args = parser.parse_args()

    entry_id  = active_audio_id
    wav_path  = audio_files[entry_id]["m4a"]
    hume_json = f"hume_ai/filtered_results/filtered/{entry_id}_filtered_emotions.json"

    df = micro_level_analysis_all(wav_path, hume_json)

    # Drop the pitch/intensity columns if you really don't want them:
    # df = df.drop(columns=["AvgPitch_Hz","AvgIntensity_dB"])

    # 1) Print & save the table
    print(f"\nSegment‑Level Table for {entry_id}:")
    print(df.to_string(index=False))

    os.makedirs("exports", exist_ok=True)
    out_xl = f"exports/segment_level_{entry_id}_soft_scores.xlsx"
    df.to_excel(out_xl, index=False)
    print("Saved Excel →", out_xl)

    # 2) Hard‑label segment confusion
    y_true = df["Hume_label"]
    y_pred = df["Praat_label"]
    print("\nClassification Report (segment):")
    print(classification_report(y_true, y_pred,
                                labels=LABEL_LIST,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_LIST)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
                cmap="Blues")
    plt.xlabel("Praat Label")
    plt.ylabel("Hume Label")
    plt.title(f"Segment Confusion ({entry_id})")
    plt.tight_layout()
    plt.show()

    # 3) Pearson r per emotion on the soft scores
    print("\nPearson r (Hume vs Praat soft‑scores) by emotion:")
    for emo in LABEL_LIST:
        h = df[f"hume_prob_{emo}"]
        p = df[f"praat_prob_{emo}"]
        mask = (~h.isna()) & (~p.isna())
        r = pearsonr(h[mask], p[mask])[0] if mask.sum()>1 else np.nan
        print(f"  {emo:>8} → r = {r:.3f}")

    # 4) Mean cosine similarity
    H = df[[f"hume_prob_{e}"   for e in LABEL_LIST]].values
    P = df[[f"praat_prob_{e}"  for e in LABEL_LIST]].values
    cos_sims = [
        np.dot(H[i],P[i])/(np.linalg.norm(H[i])*np.linalg.norm(P[i])+1e-8)
        for i in range(len(df))
    ]
    print(f"\nMean cosine similarity: {np.nanmean(cos_sims):.3f}")

    # 5) Mean JS divergence
    js = [0.5*(entropy(H[i],0.5*(H[i]+P[i])) + entropy(P[i],0.5*(H[i]+P[i])))
          for i in range(len(df))]
    print(f"Mean JS divergence: {np.nanmean(js):.3f}")

if __name__=="__main__":
    main()