## NOT USED FOR THESIS! 

import os, json, math, argparse, sys
import numpy as np, pandas as pd, parselmouth
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze
from praat_parselmouth.vocal_extract import extract_features
from utils.categorize_vocal_emotions import categorise_emotion_all_scores, categorize_emotion_table, categorize_emotion_from_vocal_markers_all
from hume_ai.hume_utils import normalize_emotions

LABEL_LIST = ['anger','fear','joy','sadness','surprise']
WINDOW     =2  # +/- 0.5s snippet

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



def get_segments(path):
    return json.load(open(path))

def micro_level_analysis_all(wav_path, segments):
    snd_full = parselmouth.Sound(wav_path)
  
    rows = []

    for seg in segments:
        t_mid = seg.get("time")
        if t_mid is None:
            continue

        # 1) pitch / intensity slice‐averages
        #    (you can keep these or drop them)
        pitch_t, pitch_v = snd_full.to_pitch().xs(), \
                           snd_full.to_pitch().selected_array["frequency"]
        pitch_v[pitch_v == 0] = np.nan
        intensity = snd_full.to_intensity()
        inten_t, inten_v = intensity.xs(), intensity.values[0]
        inten_v[inten_v == 0] = np.nan

        def seg_avg(x, v):
            m = (x >= t_mid-WINDOW) & (x <= t_mid+WINDOW)
            return np.nanmean(v[m]) if m.any() else np.nan

        row = {
            "SegmentTime":     t_mid,
            "AvgPitch_Hz":     seg_avg(pitch_t, pitch_v),
            "AvgIntensity_dB": seg_avg(inten_t, inten_v),
        }

        # 2) Hume soft‐scores, normalized
        raw_h = { emo: seg[emo] for emo in LABEL_LIST }
        row.update({ f"hume_prob_{e}": raw_h[e] for e in LABEL_LIST })
        row["Hume_label"] = max(raw_h, key=raw_h.get)

        # 3) Extract a short snippet around t_mid and re‑run your full extract_features
        #    so you get all the same features your table expects
        start = max(0, t_mid - WINDOW)
        duration = min(WINDOW*2, snd_full.get_total_duration() - start)
        snippet = snd_full.extract_part(start, duration, preserve_times=False)

        feats = extract_features(snippet)
        # 4) Praat soft‐scores via your new table function
        praat_probs = categorise_emotion_all_scores(feats)
        if len(rows) < 5:  # only for the first few
            print(f"\n--- DEBUG segment {len(rows)} at t={t_mid:.2f}s ---")
            print("Features:", feats)
            print("Praat soft‐scores:")
            for emo, sc in sorted(praat_probs.items(), key=lambda x:-x[1]):
                print(f"  {emo:>8} -> {sc:.3f}")
        row.update({ f"praat_prob_{e}": praat_probs[e] for e in LABEL_LIST })
        #row["Praat_label"] = max(praat_probs, key=praat_probs.get)
        feats = extract_features(snippet)
        probs = categorize_emotion_table(feats)
        row["Praat_label"] = max(probs, key=probs.get)

        rows.append(row)

    return pd.DataFrame(rows)

def plot_labels_over_time(df):
    # Map emotions to numeric positions
    labels = pd.concat([df["Hume_label"], df["Praat_label"]]).unique().tolist()
    label_to_int = {lbl:i for i,lbl in enumerate(labels)}

    # Convert labels to ints
    times = df["SegmentTime"].values
    hume_y  = df["Hume_label"].map(label_to_int).values
    praat_y = df["Praat_label"].map(label_to_int).values

    fig, ax = plt.subplots(figsize=(10,4))
    # plot as scatter or step
    ax.scatter(times, hume_y, marker="o", color="C1", label="Hume Label")
    ax.step( times, praat_y, where="mid", color="C0", label="Praat Label" )

    # y‐axis ticks
    ax.set_yticks(list(label_to_int.values()))
    ax.set_yticklabels(list(label_to_int.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Segment‐by‐Segment Emotion Labels: {active_audio_id}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
def plot_softcurves(df):
    plt.figure(figsize=(10, 5))
    for emo in LABEL_LIST:
        plt.plot(df["SegmentTime"], df[f"praat_prob_{emo}"],
                 label=f"Praat {emo}", linestyle="--")
        plt.plot(df["SegmentTime"], df[f"hume_prob_{emo}"],
                 label=f"Hume {emo}", alpha=0.6)
    plt.legend(loc="upper right", ncol=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Soft‐score")
    plt.title(f"Soft‐Score Curves: {active_audio_id}")
    plt.tight_layout()
    plt.show()
def main():
    entry_id  = active_audio_id
    wav_path  = audio_files[entry_id]["m4a"]
    hume_json = f"hume_ai/filtered_results/filtered/{entry_id}_filtered_emotions.json"
    segments  = json.load(open(hume_json))        # <-- load it here
    df = micro_level_analysis_all(wav_path, segments)
    #plot_softcurves(df)

    # now just plot the two label streams
   # plot_labels_over_time(df)

if __name__=="__main__":
    main()