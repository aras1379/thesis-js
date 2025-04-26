import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import parselmouth

# ensure project root for config import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files

# ---------------------------------------------
# 1) Clip-level heatmap
# ---------------------------------------------
COMPARISONS_DIR = "comparisons_rq1"
HUME_EMOS = ["anger", "fear", "joy", "sadness", "surprise"]
ACOUSTIC_FEATURES = [
    "mean_pitch_hz", "mean_intensity_db", "mean_hnr_db",
    "jitter_local", "shimmer_local"
]

def build_clip_level_df(comparisons_dir=COMPARISONS_DIR):
    records = []
    for fn in sorted(os.listdir(comparisons_dir)):
        if not fn.endswith("_vocal_vs_hume.json"): continue
        data = json.load(open(os.path.join(comparisons_dir, fn)))
        entry = data.get("entry_id", fn.split('_vocal_vs_hume.json')[0])
        vocal = data.get("vocal_features", {})
        hume  = data.get("hume_probs", {})
        row = {"entry_id": entry}
        for feat in ACOUSTIC_FEATURES:
            row[feat] = vocal.get(feat, np.nan)
        for emo in HUME_EMOS:
            row[f"hume_{emo}"] = hume.get(emo, np.nan)
        records.append(row)
    return pd.DataFrame(records).set_index('entry_id')


def plot_feature_emotion_heatmap(df):
    feat_cols = ACOUSTIC_FEATURES
    emo_cols  = [f"hume_{emo}" for emo in HUME_EMOS]
    sub = df[feat_cols + emo_cols].dropna()
    corr = sub.corr().loc[feat_cols, emo_cols]
    plt.figure(figsize=(8,5))
    sns.heatmap(corr, annot=True, center=0, cmap='RdBu_r', fmt='.2f')
    plt.title('Clip-level Pearson r: acoustic features vs Hume scores')
    plt.ylabel('Acoustic Feature')
    plt.xlabel('Hume Emotion')
    plt.tight_layout(); plt.show()

# ---------------------------------------------
# 2) Averaged cross-correlogram
# ---------------------------------------------
HUME_DIR   = "hume_ai/filtered_results/filtered"
WINDOW     = 2.5    # seconds
GRID_STEP  = 0.5
TARGET_EMO = 'joy'
MAX_LAG    = 5.0

def segment_average(times, values, t_mid, window=WINDOW):
    mask = (times >= t_mid-window) & (times <= t_mid+window)
    return np.nanmean(values[mask]) if mask.any() else np.nan

def generate_time_series(entry_id, target_emo):
    hume = json.load(open(os.path.join(HUME_DIR, f"{entry_id}_filtered_emotions.json")))
    times = np.array([seg['time'] for seg in hume])
    emo_vals = np.array([seg.get(target_emo, np.nan) for seg in hume])
    snd = parselmouth.Sound(audio_files[entry_id]['m4a'])
    p = snd.to_pitch(); t_p, v_p = p.xs(), p.selected_array['frequency']; v_p[v_p==0]=np.nan
    I = snd.to_intensity(); t_i, v_i = I.xs(), I.values[0]; v_i[v_i==0]=np.nan
    grid = np.arange(times.min(), times.max(), GRID_STEP)
    avg_p = np.array([segment_average(t_p, v_p, t) for t in grid])
    avg_i = np.array([segment_average(t_i, v_i, t) for t in grid])
    interp_fn = interp1d(times, emo_vals, kind='linear', bounds_error=False, fill_value='extrapolate')
    emo_grid = interp_fn(grid)
    return grid, avg_p, avg_i, emo_grid


import pandas as pd

def average_cross_correlogram():
    pitch_srs = []
    inten_srs = []

    # 1) For each clip, build a Series indexed by its own lags, then trim to ±MAX_LAG
    for fn in sorted(os.listdir(HUME_DIR)):
        if not fn.endswith('_filtered_emotions.json'):
            continue

        entry = fn.replace('_filtered_emotions.json','')
        grid, ap, ai, ag = generate_time_series(entry, TARGET_EMO)
        dt = grid[1] - grid[0]
        lags, cc_p = generate_crosscorr(ap, ag, dt)
        _,    cc_i = generate_crosscorr(ai, ag, dt)

        # Turn them into Series indexed by their own lag axis
        s_p = pd.Series(cc_p, index=lags)
        s_i = pd.Series(cc_i, index=lags)

        # Now slice to the desired ±MAX_LAG window
        s_p = s_p.loc[-MAX_LAG : MAX_LAG]
        s_i = s_i.loc[-MAX_LAG : MAX_LAG]

        pitch_srs.append(s_p)
        inten_srs.append(s_i)

    # 2) Build DataFrames — pandas will align all series by their index (lag)
    df_p = pd.DataFrame(pitch_srs)
    df_i = pd.DataFrame(inten_srs)

    # 3) Compute mean across clips (rows), skipping NaNs
    mean_p = df_p.mean(axis=0)
    mean_i = df_i.mean(axis=0)

    # 4) Plot
    plt.figure(figsize=(8,4))
    plt.plot(mean_p.index, mean_p.values, label='Pitch vs Hume')
    plt.plot(mean_i.index, mean_i.values, label='Intensity vs Hume')
    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Lag (s) [>0 = acoustic leads]')
    plt.ylabel('Mean cross-correlation r')
    plt.title(f'Averaged cross-correlogram ±{MAX_LAG}s for \"{TARGET_EMO}\"')
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_crosscorr(x, y, dt):
    x0, y0 = x - np.nanmean(x), y - np.nanmean(y)
    cc = correlate(x0, y0, mode='full')
    cc /= np.sqrt(np.nansum(x0**2)*np.nansum(y0**2))
    lags = np.arange(-len(x0)+1, len(x0))*dt
    return lags, cc

# ---------------------------------------------
# 3) Segment scatter + regression
# ---------------------------------------------
SEG_WINDOW = 2.5

def build_segment_df():
    rows = []
    for fn in sorted(os.listdir(HUME_DIR)):
        if not fn.endswith('_filtered_emotions.json'): continue
        entry = fn.replace('_filtered_emotions.json','')
        segs = json.load(open(os.path.join(HUME_DIR, fn)))
        snd = parselmouth.Sound(audio_files[entry]['m4a'])
        p = snd.to_pitch(); tp, vp = p.xs(), p.selected_array['frequency']; vp[vp==0]=np.nan
        I = snd.to_intensity(); ti, vi = I.xs(), I.values[0]; vi[vi==0]=np.nan
        for seg in segs:
            t = seg.get('time')
            if t is None: continue
            rows.append({
                'entry_id': entry,
                'SegmentTime': t,
                'AvgPitch_Hz': np.nanmean(vp[(tp>=t-SEG_WINDOW)&(tp<=t+SEG_WINDOW)]),
                'AvgIntensity_dB': np.nanmean(vi[(ti>=t-SEG_WINDOW)&(ti<=t+SEG_WINDOW)]),
                **{f'Hume_{emo}': seg.get(emo, np.nan) for emo in HUME_EMOS}
            })
    return pd.DataFrame(rows)


def scatter_regression(df):
    for emo in HUME_EMOS:
        col = f'Hume_{emo}'
        sns.lmplot(x='AvgPitch_Hz',    y=col, data=df, height=4, aspect=1.2, scatter_kws={'alpha':0.3}, ci=95)
        plt.title(f'Pitch vs Hume {emo.title()}'); plt.tight_layout(); plt.show()
        sns.lmplot(x='AvgIntensity_dB',y=col, data=df, height=4, aspect=1.2, scatter_kws={'alpha':0.3}, ci=95)
        plt.title(f'Intensity vs Hume {emo.title()}'); plt.tight_layout(); plt.show()

# ---------------------------------------------
# 4) Composite acoustic indices
# ---------------------------------------------
def compute_and_plot_composite(df):
    """
    Build simple freq/amp composite indices using only the features available
    in df (AvgPitch_Hz, AvgIntensity_dB, and jitter_local/shimmer_local if present),
    then correlate each index with every Hume emotion across segments.
    """
    df_z = df.copy()
    # Determine which features we actually have
    freq_feats = ['AvgPitch_Hz']
    amp_feats = ['AvgIntensity_dB']
    for col in ['jitter_local', 'shimmer_local']:
        if col in df_z.columns:
            amp_feats.append(col)
    # z-score each feature
    for feat in freq_feats + amp_feats:
        zcol = feat + '_z'
        df_z[zcol] = (df_z[feat] - df_z[feat].mean()) / df_z[feat].std()
    # composite indices
    df_z['freq_index'] = np.mean([df_z[f + '_z'] for f in freq_feats], axis=0)
    df_z['amp_index']  = np.mean([df_z[f + '_z'] for f in amp_feats], axis=0)
    # correlate with each emotion
    results = []
    for emo in HUME_EMOS:
        col = f'Hume_{emo}'
        sub = df_z.dropna(subset=['freq_index', 'amp_index', col])
        r_freq, p_freq = pearsonr(sub['freq_index'], sub[col]) if len(sub) > 1 else (np.nan, np.nan)
        r_amp,  p_amp  = pearsonr(sub['amp_index'],  sub[col]) if len(sub) > 1 else (np.nan, np.nan)
        results.append({'emotion': emo, 'freq_r': r_freq, 'freq_p': p_freq,
                        'amp_r': r_amp,   'amp_p': p_amp})
    comp_df = pd.DataFrame(results).set_index('emotion')
    print("Composite index correlations:")
    print(comp_df)
    # bar plot of r-values
    comp_df[['freq_r','amp_r']].plot.bar(figsize=(6,4))
    plt.title('Composite freq/amp index vs Hume emotion r')
    plt.ylabel('Pearson r')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df_clips = build_clip_level_df()
    #plot_feature_emotion_heatmap(df_clips)
    #average_cross_correlogram()
    df_segs = build_segment_df()
    #scatter_regression(df_segs)
    compute_and_plot_composite(df_segs)
