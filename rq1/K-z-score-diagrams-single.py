## USE
# vocal features over time single clip 
# z score fluct single clip

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import statsmodels.formula.api as smf
from config_rq1 import EXPORT_DIR, INPUT_DIR_V3, PLOT_DIR, EMO_LABELS

# Project utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import active_audio_id, audio_files

from utils.data_utils import plot_and_save
from utils.parselmouth_utils import segment_and_extract_features

# ---------------------------------------------
# Constants
# ---------------------------------------------
COMPARISONS_DIR = INPUT_DIR_V3
EXPORTS_DIR     = EXPORT_DIR
PLOT_DIR = PLOT_DIR
SEGMENT_LENGTH  = 2.5  # seconds
HUME_DIR       = "hume_ai/filtered_results/filtered"
LABELS         = EMO_LABELS
# ---------------------------------------------


def analyze_segment_statistics(z_scores, hume, vocal_feats, hume_emotions, clip_id, export_dir=EXPORTS_DIR):
    """
    Pearson r between vocal feat z_score and each hume_emotion
    High-vs-Low t-test (top 30% vs bottom 70% of emotion prob)
    Linear trend of each vocal feat over time
    """
    os.makedirs(export_dir, exist_ok=True)
    # Correlations
    corr_rows = []
    for feat in vocal_feats:
        for emo in hume_emotions:
            merged = pd.merge_asof(
                z_scores.sort_values("Segment_Start"),
                hume.sort_values("Segment_Start"),
                on="Segment_Start",
                direction="nearest"
            ).dropna(subset=[feat, emo])
            if len(merged) > 1:
                r, p = pearsonr(merged[feat], merged[emo])
            else:
                r, p = np.nan, np.nan
            corr_rows.append({
                "Feature": feat,
                "Emotion": emo,
                "Pearson_r": round(r,3),
                "p-value": round(p,4),
                "Significant": "Yes" if p<0.05 else "No"
            })
    df_corr = pd.DataFrame(corr_rows)
    fn = os.path.join(export_dir, f"corr_segment_feature_vs_hume_{clip_id}.xlsx")
    df_corr.to_excel(fn, index=False)
    print(f"Saved correlations → {fn}")

    # High vs Low t-tests
    t_rows = []
    for feat in vocal_feats:
        for emo in hume_emotions:
            merged = pd.merge_asof(
                z_scores.sort_values("Segment_Start"),
                hume.sort_values("Segment_Start"),
                on="Segment_Start",
                direction="nearest"
            ).dropna(subset=[feat, emo])
            if len(merged) > 1:
                thresh = merged[emo].quantile(0.7)
                high = merged.loc[merged[emo]>=thresh, feat]
                low  = merged.loc[merged[emo]< thresh, feat]
                t, p = ttest_ind(high, low, nan_policy="omit")
            else:
                t, p = np.nan, np.nan
            t_rows.append({
                "Feature": feat,
                "Emotion": emo,
                "Group": "High vs Low",
                "t-statistic": round(t,3),
                "p-value": round(p,4),
                "Significant": "Yes" if p<0.05 else "No"
            })
    df_ttest = pd.DataFrame(t_rows)
    fn = os.path.join(export_dir, f"ttest_highlow_segment_feature_vs_emotion_{clip_id}.xlsx")
    df_ttest.to_excel(fn, index=False)
    print(f"Saved high/low t-tests → {fn}")

    # Linear trends
    trend_rows = []
    for feat in vocal_feats:
        formula = f"{feat} ~ Segment_Start"
        model   = smf.ols(formula, data=z_scores).fit()
        coef    = model.params["Segment_Start"]
        pval    = model.pvalues["Segment_Start"]
        trend_rows.append({
            "Feature": feat,
            "Slope": round(coef, 3),
            "p-value": round(pval, 4),
            "Significant": "Yes" if pval < 0.05 else "No"
        })
    df_trend = pd.DataFrame(trend_rows)
    fn = os.path.join(export_dir, f"trend_segment_feature_over_time_{clip_id}.xlsx")
    df_trend.to_excel(fn, index=False)
    print(f"Saved linear trends  {fn}")

    return df_corr, df_ttest, df_trend


    
# ---------------------------------------------
"""
Prints single clip vocal feats 
+ average mean value 
"""
def analyze_clip_segments_vocal():
    clip_id   = active_audio_id
    audio_path = audio_files[clip_id]['wav']

    # 1) extract segments
    df = segment_and_extract_features(audio_path, SEGMENT_LENGTH)

    # 2) z‐score against baseline
    baseline = pd.read_excel(os.path.join(EXPORTS_DIR, "vocal_features_baseline.xlsx"), index_col=0)
    feat_cols = [c for c in df.columns if c != "Segment_Start"]
    z_scores = (df[feat_cols] - baseline["Baseline_Mean"]) / baseline["Baseline_Std"]
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

    plot_and_save(fig, f"{PLOT_DIR}/vocal_time_{active_audio_id}")

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

    plot_and_save(fig, f"{PLOT_DIR}/zscore_fluctuations_{active_audio_id}")


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


def analyze_combined_zscores_and_hume(
    vocal_feats=None,
    hume_emotions=None,
    segment_length=SEGMENT_LENGTH,
    save_idx: int = 1
):
    """
    Plot vocal‐feature z‐scores and Hume emotion curves on the same time axis.
    Saves to plots/combined_zscore_hume_<clip_id>_<save_idx>.pdf
    """

    clip_id  = active_audio_id
    wav_path = audio_files[clip_id]['wav']

    # default feature/emotion sets
    if vocal_feats   is None: vocal_feats   = ['mean_pitch_hz','mean_intensity_db','mean_hnr_db','jitter_local','shimmer_local']
    if hume_emotions is None: hume_emotions = LABELS

    vocal_palette = sns.color_palette("pastel", n_colors=len(vocal_feats))
    hume_palette  = sns.color_palette("deep",   n_colors=len(hume_emotions))

    # --- build z-scores ---
    df = segment_and_extract_features(wav_path, segment_length)

    baseline = pd.read_excel(os.path.join(EXPORTS_DIR, "vocal_features_baseline.xlsx"), index_col=0)
    z = (df[vocal_feats] - baseline.loc[vocal_feats, 'Baseline_Mean']) \
        / baseline.loc[vocal_feats, 'Baseline_Std']
    z['Segment_Start'] = df['Segment_Start']

    # --- load Hume probabilities ---
    hume_fn = next(
        f for f in os.listdir(HUME_DIR)
        if f.startswith(clip_id) and f.endswith('_filtered_emotions.json')
    )
    hume = pd.DataFrame(json.load(open(os.path.join(HUME_DIR, hume_fn))))
    hume = hume.rename(columns={'time':'Segment_Start'})

    merged = pd.merge_asof(
        z.sort_values('Segment_Start'),
        hume.sort_values('Segment_Start'),
        on='Segment_Start',
        direction='nearest',
        tolerance=segment_length/2
    ).dropna(subset=hume_emotions)

    # --- plot ---
    fig, ax1 = plt.subplots(figsize=(14,6))

    # Left axis: Vocal Z-scores
    for feat, col in zip(vocal_feats, vocal_palette):
        sns.lineplot(
            data=z, x='Segment_Start', y=feat,
            label=feat.replace('_',' ').title(),
            color=col, ax=ax1
        )
    ax1.set_ylabel("Vocal Feature Z-Score")
    ax1.set_xlabel("Time (s)")

    # Right axis: Hume
    ax2 = ax1.twinx()
    for emo, col in zip(hume_emotions, hume_palette):
        sns.lineplot(
            data=merged, x='Segment_Start', y=emo,
            label=emo.title(),
            color=col, linestyle='--', ax=ax2
        )
    ax2.set_ylabel("Hume Emotion Probability")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    vf_labels  = [lbl for lbl in l1 if lbl in [f.replace('_',' ').title() for f in vocal_feats]]
    vf_handles = [h for h,lbl in zip(h1,l1) if lbl in vf_labels]
    ax1.legend(vf_handles, vf_labels,
               loc='upper right', title="Vocal Features", fontsize='small')

    he_labels   = [lbl for lbl in l2 if lbl in [e.title() for e in hume_emotions]]
    he_handles  = [h for h,lbl in zip(h2,l2) if lbl in he_labels]
    ax2.legend(he_handles, he_labels,
               loc='upper left', title="Hume Emotions", fontsize='small')

    ax1.set_title(f"{clip_id}: Vocal Z-Scores & Hume Over Time")

    plt.tight_layout()
    save_name = f"combined_zscore_hume_{clip_id}_{save_idx}"
    plot_and_save(fig, os.path.join(PLOT_DIR, save_name))
    #plt.close(fig)

    plt.show()


# ---------------------------------------------
if __name__ == "__main__":
    generate_vocal_feature_baseline()
    
    #analyze_clip_segments_vocal()
    
    #analyze_combined_zscores_and_hume(save_idx=3)
    # analyze_combined_zscores_and_hume(
    #     vocal_feats=['mean_pitch_hz', 'mean_intensity_db'],
    #     hume_emotions=['joy'],
    #     save_idx=4
    # )
    # analyze_combined_zscores_and_hume(
    #     vocal_feats=['mean_pitch_hz', 'mean_intensity_db', 'mean_hnr_db'],
    #     hume_emotions=['sadness'],
    #     save_idx=6
    # )
    analyze_combined_zscores_and_hume(
        vocal_feats=['mean_pitch_hz', 'mean_hnr_db'],
        hume_emotions=['sadness'],
        save_idx=7
    )
    
 
   