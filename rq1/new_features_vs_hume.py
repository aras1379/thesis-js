## USE

import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
import parselmouth

# Statsmodels for ANOVA and Tukey (optional)
# You can replace these with SciPy or pingouin if preferred
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Scikit-learn for classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Project utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import plot_and_save
from config import audio_files
from utils.categorize_vocal_emotions import FEATURE_STATS

# ---------------------------------------------
# Constants & Settings
# ---------------------------------------------
COMPARISONS_DIR   = "comparisons_rq1"
HUME_DIR         = "hume_ai/filtered_results/filtered"
LABELS           = ['anger','fear','joy','sadness','surprise']
WINDOW           = 2.5    # seconds for segment averaging
GRID_STEP        = 0.5
TARGET_EMO       = 'joy' #change for some functions ! 
MAX_LAG          = 5.0
SEG_WINDOW       = 2.5
# acoustic features that were extracted in comparisons_rq1 JSONs
ACOUSTIC_FEATURES = [
    "mean_pitch_hz", "mean_intensity_db", "mean_hnr_db",
    "jitter_local", "shimmer_local"
]
HUME_EMOS        = LABELS

# ---------------------------------------------
# 1) Clip-level DataFrame & Plots
# ---------------------------------------------
def build_clip_level_df(comparisons_dir=COMPARISONS_DIR):
    records = []
    for fn in sorted(os.listdir(comparisons_dir)):
        if not fn.endswith("_vocal_vs_hume.json"): continue
        data = json.load(open(os.path.join(comparisons_dir, fn)))
        entry = data.get("entry_id")
        vf    = data.get("vocal_features", {})
        ps    = data.get("praat_scores", {})
        hs    = data.get("hume_probs", {})
        row = {"entry_id": entry}
        # acoustic markers
        for feat in ACOUSTIC_FEATURES:
            row[feat] = vf.get(feat, np.nan)
        # soft scores
        for emo in HUME_EMOS:
            row[f"hume_{emo}"] = hs.get(emo, np.nan)
            row[f"praat_{emo}"] = ps.get(emo, np.nan)
        records.append(row)
    return pd.DataFrame(records).set_index('entry_id')


def plot_heatmap(df, row_feats, col_feats, title, save_dir="plots"):
    corr = df.corr().loc[row_feats, col_feats]
    fig, ax = plt.subplots(figsize=(len(col_feats), len(row_feats)))
    sns.heatmap(corr, annot=True, center=0, cmap='RdBu_r', fmt='.2f', ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    filename = f"{save_dir}/{title.lower().replace(' ', '_')}"
    plot_and_save(fig, filename)



def run_clip_level_plots(df):
    fv = ACOUSTIC_FEATURES
    hv = [f'hume_{e}' for e in HUME_EMOS]
    pv = [f'praat_{e}' for e in HUME_EMOS]
    # correlation heatmaps
    plot_heatmap(df, fv, hv, 'Vocal features vs Hume correlations')
    plot_heatmap(df, fv, pv, 'Vocal features vs Praat correlations')
    # violin distributions
    df['hume_label']  = df[hv].idxmax(axis=1).str.replace('hume_','')
    # for feat in fv:
    #     plt.figure(figsize=(6,4))
    #     sns.violinplot(x='hume_label', y=feat, data=df, inner='box')
    #     plt.title(f'{feat} by Hume label'); plt.tight_layout(); plt.show()
    # ANOVA + Tukey HSD
    for feat in fv:
        df['lab'] = df['hume_label']
        model = ols(f"{feat} ~ C(lab)", data=df).fit()
        print(f"ANOVA for {feat}:\n", sm.stats.anova_lm(model, typ=2))
        print(pairwise_tukeyhsd(df[feat], df['lab']))
        
def summarize_anova_results(df):
    summary = []
    fv = ACOUSTIC_FEATURES
    df['hume_label'] = df[[f'hume_{e}' for e in HUME_EMOS]].idxmax(axis=1).str.replace('hume_', '')

    for feat in fv:
        model = ols(f"{feat} ~ C(hume_label)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table["PR(>F)"][0]

        # Check if significant
        significant = "Yes" if p_value < 0.05 else "No"

        summary.append({
            "Feature": feat,
            "ANOVA p-value": round(p_value, 4),
            "Significant Differences?": significant
        })

    summary_df = pd.DataFrame(summary)
    print("\nANOVA Summary Across Acoustic Features:")
    print(summary_df)

    # Optional: Save to Excel or CSV for easy copy-paste into thesis
    summary_df.to_excel("exports/anova_summary.xlsx", index=False)

    return summary_df

# ---------------------------------------------
# 2) Cross-correlation
# ---------------------------------------------
# def segment_average(times, values, t_mid, window=WINDOW):
#     mask = (times>=t_mid-window)&(times<=t_mid+window)
#     return np.nanmean(values[mask]) if mask.any() else np.nan


# def generate_time_series(entry_id, target_emo=TARGET_EMO):
#     segs = json.load(open(os.path.join(HUME_DIR, f"{entry_id}_filtered_emotions.json")))
#     times = np.array([s['time'] for s in segs])
#     emo   = np.array([s.get(target_emo, np.nan) for s in segs])
#     snd   = parselmouth.Sound(audio_files[entry_id]['m4a'])
#     p     = snd.to_pitch(); t_p, v_p = p.xs(), p.selected_array['frequency']; v_p[v_p==0]=np.nan
#     I     = snd.to_intensity(); t_i, v_i = I.xs(), I.values[0]; v_i[v_i==0]=np.nan
#     grid  = np.arange(times.min(), times.max(), GRID_STEP)
#     avg_p = np.array([segment_average(t_p,v_p,t) for t in grid])
#     avg_i = np.array([segment_average(t_i,v_i,t) for t in grid])
#     fn    = interp1d(times, emo, kind='linear', fill_value='extrapolate')
#     return grid, avg_p, avg_i, fn(grid)


# def generate_crosscorr(x,y,dt):
#     x0,y0 = x-np.nanmean(x), y-np.nanmean(y)
#     cc    = correlate(x0,y0,mode='full')
#     cc   /= np.sqrt(np.nansum(x0**2)*np.nansum(y0**2))
#     lags  = np.arange(-len(x0)+1,len(x0))*dt
#     return lags, cc


# def average_cross_correlogram():
#     all_p, all_i = [], []
#     l2 = None
#     for fn in os.listdir(HUME_DIR):
#         if not fn.endswith('_filtered_emotions.json'): continue
#         eid = fn.replace('_filtered_emotions.json','')
#         grid, ap, ai, eg = generate_time_series(eid)
#         dt = grid[1] - grid[0]
#         l, cp = generate_crosscorr(ap, eg, dt)
#         _, ci = generate_crosscorr(ai, eg, dt)
#         mask = (l >= -MAX_LAG) & (l <= MAX_LAG)
#         arr_p, arr_i = cp[mask], ci[mask]
#         all_p.append(arr_p)
#         all_i.append(arr_i)
#         l2 = l[mask]  # last one; all arr_p should have been padded/truncated to this length

#     # pad/truncate to common length
#     maxlen = len(l2)
#     padded_p = []
#     padded_i = []
#     for arr in all_p:
#         pad = np.full(maxlen, np.nan)
#         pad[:len(arr)] = arr[:maxlen]
#         padded_p.append(pad)
#     for arr in all_i:
#         pad = np.full(maxlen, np.nan)
#         pad[:len(arr)] = arr[:maxlen]
#         padded_i.append(pad)

#     mean_p = np.nanmean(padded_p, axis=0)
#     mean_i = np.nanmean(padded_i, axis=0)

#     plt.plot(l2, mean_p, label='Pitch vs Hume')
#     plt.plot(l2, mean_i, label='Intensity vs Hume')
#     plt.axvline(0, ls='--', color='k')
#     plt.title(f'Cross-corr for \"{TARGET_EMO}\"')
#     plt.legend(); plt.tight_layout(); plt.show()

# ---------------------------------------------
# 3) Segment-level regression scatter
# ---------------------------------------------
def build_segment_df():
    rows=[]
    for fn in os.listdir(HUME_DIR):
        if not fn.endswith('_filtered_emotions.json'): continue
        eid = fn.replace('_filtered_emotions.json','')
        segs = json.load(open(os.path.join(HUME_DIR, fn)))
        snd = parselmouth.Sound(audio_files[eid]['m4a'])
        p   = snd.to_pitch(); tp, vp = p.xs(), p.selected_array['frequency']; vp[vp==0]=np.nan
        I   = snd.to_intensity(); ti, vi = I.xs(), I.values[0]; vi[vi==0]=np.nan
        for s in segs:
            t = s['time']
            rows.append({
                'SegmentTime': t,
                'AvgPitch_Hz': np.nanmean(vp[(tp>=t-SEG_WINDOW)&(tp<=t+SEG_WINDOW)]),
                'AvgIntensity_dB': np.nanmean(vi[(ti>=t-SEG_WINDOW)&(ti<=t+SEG_WINDOW)]),
                **{f'hume_{e}': s.get(e, np.nan) for e in LABELS}
            })
    return pd.DataFrame(rows)


# def scatter_regression(df):
#     for emo in LABELS:
#         sns.lmplot(x='AvgPitch_Hz', y=f'hume_{emo}', data=df)
#         plt.title(f'Pitch vs Hume {emo}')
#         plt.tight_layout(rect=(0, 0, 1, 0.98))
#         plt.show()
    
#         sns.lmplot(x='AvgIntensity_dB', y=f'hume_{emo}', data=df)
#         plt.title(f'Intensity vs Hume {emo}')
#         plt.tight_layout(rect=(0, 0, 1, 0.98))
#         plt.show()
      

# ---------------------------------------------
# 4) Composite indices

# ---------------------------------------------
def compute_and_plot_composite(df, save_dir="plots"):
    dfz = df.copy()
    dfz['Pitch_z'] = (dfz['AvgPitch_Hz'] - dfz['AvgPitch_Hz'].mean()) / dfz['AvgPitch_Hz'].std()
    dfz['Intensity_z'] = (dfz['AvgIntensity_dB'] - dfz['AvgIntensity_dB'].mean()) / dfz['AvgIntensity_dB'].std()

    res = []
    for emo in LABELS:
        sub = dfz.dropna(subset=[f'hume_{emo}', 'Pitch_z', 'Intensity_z'])
        pitch_r, pitch_p = pearsonr(sub['Pitch_z'], sub[f'hume_{emo}'])
        int_r, int_p     = pearsonr(sub['Intensity_z'], sub[f'hume_{emo}'])
        res.append({
            'Emotion': emo.title(),
            'Pitch_r': round(pitch_r, 3),
            'Pitch_p': round(pitch_p, 4),
            'Intensity_r': round(int_r, 3),
            'Intensity_p': round(int_p, 4)
        })

    cdf = pd.DataFrame(res).set_index('Emotion')

    # Plot only r-values
    fig, ax = plt.subplots(figsize=(10, 6))
    cdf[['Pitch_r', 'Intensity_r']].plot.bar(ax=ax)

    ax.set_title("Composite Correlations: Pitch & Intensity vs Emotions")
    ax.set_ylabel("Pearson r")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    filename = f"{save_dir}/composite_correlations"
    plot_and_save(fig, filename)

    # Export table with r and p-values
    export_path = "exports/composite_correlations_table.xlsx"
    cdf.to_excel(export_path)
    print(f"Saved correlation table: {export_path}")

    return cdf



# ---------------------------------------------
# 5) Classifier & feature importances
# ---------------------------------------------
def run_classifier(df):
    df = df.dropna(subset=ACOUSTIC_FEATURES + [f'hume_{e}' for e in LABELS])
    df['label'] = df[[f'hume_{e}' for e in LABELS]].idxmax(axis=1).str.replace('hume_','')
    X = df[ACOUSTIC_FEATURES].values
    y = LabelEncoder().fit_transform(df['label'])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    acc = cross_val_score(clf, X, y, cv=cv)
    print(f"RF acc: {acc.mean():.3f} ± {acc.std():.3f}")
    clf.fit(X, y)
    imps = pd.Series(clf.feature_importances_, index=ACOUSTIC_FEATURES).sort_values()
    print(imps)
    return imps

# ---------------------------------------------
# 6) Compare to Ekberg
# ---------------------------------------------
def compare_ekberg(imps, save_dir="plots"):
    """
    Build a max-|z| importance index for each feature based on Ekberg et al.'s
    published means and SDs (in FEATURE_STATS), then correlate with model importances.
    """
    feat_map = {
        'pitch':    'mean_pitch_hz',
        'loudness': 'mean_intensity_db',
        'hnr':      'mean_hnr_db',
        'jitter':   'jitter_local',
        'shimmer':  'shimmer_local',
    }
    ek = {}
    for stat_key, emo_stats in FEATURE_STATS.items():
        feat_name = feat_map.get(stat_key)
        if feat_name is None or feat_name not in imps.index:
            continue
        z_vals = [abs(mu/sd) for mu, sd in emo_stats.values() if sd > 0]
        ek[feat_name] = max(z_vals) if z_vals else 0.0

    edf = pd.Series(ek).rename('max_z')
    comp = edf.to_frame().join(imps.rename('imp'))
    rho, p = spearmanr(comp['max_z'], comp['imp'], nan_policy='omit')
    print(f"Spearman ρ = {rho:.2f}, p = {p:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(comp['max_z'], comp['imp'])

    # Lägg till textetiketter
    for feat in comp.index:
        ax.text(comp.loc[feat,'max_z'], comp.loc[feat,'imp'], feat)

    ax.set_xlabel('Ekberg max |z|')
    ax.set_ylabel('Model Feature Importance')
    ax.set_title(f'Spearman Correlation: ρ = {rho:.2f}, p = {p:.3f}')
    plt.tight_layout()

    filename = f"{save_dir}/ekberg_vs_importance"
    plot_and_save(fig, filename)

    return comp  # Om du vill analysera vidare


# ---------------------------------------------
# 7) Full heatmap of all vocal_features
# ---------------------------------------------
def build_clip_full_df(comparisons_dir=COMPARISONS_DIR):
    records = []
    for fn in sorted(os.listdir(comparisons_dir)):
        if not fn.endswith('_vocal_vs_hume.json'): continue
        d = json.load(open(os.path.join(comparisons_dir, fn)))
        eid = d['entry_id']
        vf  = d['vocal_features']
        hs  = d['hume_probs']
        row = {'entry_id': eid}
        for k, v in vf.items():
            if isinstance(v, dict):
                for sub, val in v.items(): row[f'{k}_{sub}'] = val
            else:
                row[k] = v
        for emo in LABELS:
            row[f'hume_{emo}'] = hs.get(emo, np.nan)
        records.append(row)
    return pd.DataFrame(records).set_index('entry_id')

# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == '__main__':
    # 1) clip-level analysis
    df_clips = build_clip_level_df()
    run_clip_level_plots(df_clips)
    summarize_anova_results(df_clips)
    # 2) cross-correlation
   # average_cross_correlogram()
    # 3) segment-level regression
    df_segs = build_segment_df()
    #scatter_regression(df_segs)
    # 4) composite indices
    compute_and_plot_composite(df_segs)
    # 5) classifier
    imps = run_classifier(df_clips)
    # 6) compare to Ekberg
    compare_ekberg(imps)
    # 7) full feature vs Hume heatmap
    df_full = build_clip_full_df()
    plot_heatmap(df_full,
                 [c for c in df_full.columns if not c.startswith('hume_')],
                 [f'hume_{e}' for e in LABELS],
                 'Full feature vs Hume heatmap')
