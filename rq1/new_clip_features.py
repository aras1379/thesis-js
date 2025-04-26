## USE
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import parselmouth
import json

# Project utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import active_audio_id, audio_files

from utils.data_utils import plot_and_save
from praat_parselmouth.vocal_extract import extract_features

# ---------------------------------------------
# Constants
# ---------------------------------------------
COMPARISONS_DIR = "comparisons_rq1"
EXPORTS_DIR     = "exports"
SEGMENT_LENGTH  = 5  # seconds

# ---------------------------------------------
def analyze_clip_average():
    audio_path = audio_files[active_audio_id]['wav']
    features = extract_features(audio_path)

    flat_features = features.copy()
    formants = flat_features.pop("formants_hz")
    for k, v in formants.items():
        flat_features[f"Formant_{k}_Hz"] = v

    df = pd.DataFrame([flat_features])
    print("\nAverage Vocal Features for Clip:")
    print(df.T)

    export_path = os.path.join(EXPORTS_DIR, f"{active_audio_id}_average_features.xlsx")
    df.to_excel(export_path, index=False)

    plt.figure(figsize=(10,5))
    df_plot = df.drop(columns=[col for col in df.columns if "Formant" in col])
    df_plot.T.plot(kind='bar', legend=False)
    plt.title(f"Average Vocal Features: {active_audio_id}")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
def analyze_clip_segments():
    audio_path = audio_files[active_audio_id]['wav']
    snd = parselmouth.Sound(audio_path)
    duration = snd.get_total_duration()

    segments = []
    t = 0.0
    while t < duration:
        snippet_duration = min(SEGMENT_LENGTH, duration - t)
        snippet = snd.extract_part(t, t + snippet_duration, preserve_times=False)
        feats = extract_features(snippet)
        flat_feats = feats.copy()
        formants = flat_feats.pop("formants_hz")
        for k, v in formants.items():
            flat_feats[f"Formant_{k}_Hz"] = v
        flat_feats["Segment_Start"] = round(t, 2)
        segments.append(flat_feats)
        t += SEGMENT_LENGTH  # Always step by SEGMENT_LENGTH

    df = pd.DataFrame(segments)
    baseline = pd.read_excel(os.path.join(EXPORTS_DIR, "vocal_features_baseline.xlsx"), index_col=0)

    feature_cols = [col for col in df.columns if col != "Segment_Start"]
    z_scores = (df[feature_cols] - baseline["Baseline_Mean"]) / baseline["Baseline_Std"]
    z_scores["Segment_Start"] = df["Segment_Start"]

    z_export_path = os.path.join(EXPORTS_DIR, f"{active_audio_id}_segment_zscores.xlsx")
    z_scores.to_excel(z_export_path, index=False)
    print(f"\nZ-scores saved to {z_export_path}")

    print("\nSegmented Vocal Features:")
    print(df)

    export_path = os.path.join(EXPORTS_DIR, f"{active_audio_id}_segmented_features.xlsx")
    df.to_excel(export_path, index=False)

    fig, ax = plt.subplots(figsize=(12,6)) 
    for feature in ["mean_pitch_hz", "mean_intensity_db", "mean_hnr_db"]:
        sns.lineplot(data=df, x="Segment_Start", y=feature, label=feature, ax=ax)  # Använd ax

    ax.set_xlabel("Time (s)")
    ax.set_title(f"Vocal Features Over Time: {active_audio_id}")
    ax.legend()
    plt.tight_layout()

    save_dir = "plots"
    plot_and_save(fig, f"{save_dir}/vocal_time_{active_audio_id}")

    interpret_vocal_zscores(z_scores)

# ---------------------------------------------
def interpret_vocal_zscores(zscore_df):
    flagged_segments = []

    for idx, row in zscore_df.iterrows():
        cues = []
        if row['mean_pitch_hz'] > 1 and row['mean_intensity_db'] > 1:
            cues.append("High Arousal (Anger/Joy/Surprise)")
        elif row['mean_pitch_hz'] < -1 and row['mean_intensity_db'] < -1:
            cues.append("Low Arousal (Sadness)")

        if row['jitter_local'] > 1.5 or row['shimmer_local'] > 1.5:
            cues.append("Vocal Instability (Surprise/Nervousness)")

        if cues:
            flagged_segments.append({
                "Segment_Start": row['Segment_Start'],
                "Cues": "; ".join(cues)
            })

    flagged_df = pd.DataFrame(flagged_segments)
    print("\nFlagged Segments with Emotional Cues:")
    print(flagged_df)

    flagged_export = os.path.join(EXPORTS_DIR, f"{active_audio_id}_flagged_segments.xlsx")
    flagged_df.to_excel(flagged_export, index=False)
    print(f"\nFlagged segments saved to {flagged_export}")

    features_to_plot = ['mean_pitch_hz', 'mean_intensity_db', 'jitter_local', 'shimmer_local']
    fig, ax = plt.subplots(figsize=(12,6))   # Skapa fig och ax

    for feature in features_to_plot:
        sns.lineplot(data=zscore_df, x='Segment_Start', y=feature, label=feature, ax=ax)  # Peka på axeln

    # Lägg till horisontella linjer på samma axel
    ax.axhline(1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='blue', linestyle='--', alpha=0.5)

    # Sätt labels och titel via ax
    ax.set_title(f"Z-Score Fluctuations Over Time: {active_audio_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Z-Score")
    ax.legend()

    plt.tight_layout()

    # Spara och visa
    save_dir = "plots"
    plot_and_save(fig, f"{save_dir}/zscore_fluctuations_{active_audio_id}")


# ---------------------------------------------
def generate_vocal_feature_baseline():
    records = []
    for fn in os.listdir(COMPARISONS_DIR):
        if fn.endswith("_vocal_vs_hume.json"):
            data = json.load(open(os.path.join(COMPARISONS_DIR, fn)))
            vf = data.get("vocal_features", {})
            flat = vf.copy()
            formants = flat.pop("formants_hz", {})
            for k, v in formants.items():
                flat[f"Formant_{k}_Hz"] = v
            records.append(flat)

    df = pd.DataFrame(records)
    baseline = df.agg(['mean', 'std']).T.rename(columns={'mean': 'Baseline_Mean', 'std': 'Baseline_Std'})
    baseline.to_excel(os.path.join(EXPORTS_DIR, "vocal_features_baseline.xlsx"))
    print("\nBaseline saved to exports/vocal_features_baseline.xlsx")
    return baseline

# ---------------------------------------------
if __name__ == "__main__":
    generate_vocal_feature_baseline()
    analyze_clip_average()
    analyze_clip_segments()