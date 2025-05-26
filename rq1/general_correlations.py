## USE
## Change to seperate negative and positive clips 

## Heatmaps 
## composite correlation 

## anova in excel 
import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import parselmouth

from statsmodels.formula.api import ols
import statsmodels.api as sm

# Project utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import plot_and_save
from config import audio_files
from config_rq1 import INPUT_DIR_V3, PLOT_DIR, EMO_LABELS, EXPORT_DIR

from common_functions import( infer_sentiment, group_by_sentiment_files, filter_by_sentiment, SENTIMENTS)


COMPARISONS_DIR = INPUT_DIR_V3
EXPORT_DIR = EXPORT_DIR
PLOT_DIR = PLOT_DIR
HUME_DIR       = "hume_ai/filtered_results/filtered"
LABELS         = EMO_LABELS
ACOUSTIC_FEATURES = [
    "mean_pitch_st", "mean_intensity_db", "mean_hnr_db",
    "jitter_local", "shimmer_local"
]
SEG_WINDOW     = 2.25


# ---------------------------------------------
# DataFrame builders
# ---------------------------------------------
def build_clip_level_df():
    records = []
    for fn in sorted(os.listdir(COMPARISONS_DIR)):
        if not fn.endswith('_vocal_vs_hume.json'): continue
        path = os.path.join(COMPARISONS_DIR, fn)
        data = json.load(open(path, encoding='utf-8'))
        entry_id = data.get('entry_id')
        sentiment = infer_sentiment(fn)
        vf = data.get('vocal_features', {})
        ps = data.get('praat_scores', {})
        hs = data.get('hume_probs', {})
        row = {'entry_id': entry_id, 'sentiment': sentiment}
        for feat in ACOUSTIC_FEATURES:
            row[feat] = vf.get(feat, np.nan)
        for emo in LABELS:
            row[f'hume_{emo}'] = hs.get(emo, np.nan)
            row[f'praat_{emo}'] = ps.get(emo, np.nan)
        records.append(row)
    return pd.DataFrame(records).set_index('entry_id')


def build_segment_df():
    rows = []
    for fn in os.listdir(HUME_DIR):
        if not fn.endswith('_filtered_emotions.json'): continue
        entry_id = fn.replace('_filtered_emotions.json','')
        # find the matching comparison filename to get sentiment
        comp_fn = next((c for c in os.listdir(COMPARISONS_DIR) if entry_id in c and c.endswith('_vocal_vs_hume.json')), '')
        sentiment = infer_sentiment(comp_fn)
        audio_path = audio_files.get(entry_id, {}).get('m4a', '')
        segs = json.load(open(os.path.join(HUME_DIR, fn)))
        snd = parselmouth.Sound(audio_path)
        p = snd.to_pitch(); tp, vp = p.xs(), p.selected_array['frequency']; vp[vp==0]=np.nan
        I = snd.to_intensity(); ti, vi = I.xs(), I.values[0]; vi[vi==0]=np.nan
        for s in segs:
            t = s['time']
            rows.append({
                'entry_id': entry_id,
                'sentiment': sentiment,
                'SegmentTime': t,
                'AvgPitch_Hz': np.nanmean(vp[(tp>=t-SEG_WINDOW)&(tp<=t+SEG_WINDOW)]),
                'AvgIntensity_dB': np.nanmean(vi[(ti>=t-SEG_WINDOW)&(ti<=t+SEG_WINDOW)]),
                **{f'hume_{e}': s.get(e, np.nan) for e in LABELS}
            })
    return pd.DataFrame(rows)


def build_clip_full_df():
    records = []
    for fn in sorted(os.listdir(COMPARISONS_DIR)):
        if not fn.endswith('_vocal_vs_hume.json'): continue
        path = os.path.join(COMPARISONS_DIR, fn)
        data = json.load(open(path, encoding='utf-8'))
        entry_id = data.get('entry_id')
        sentiment = infer_sentiment(fn)
        vf = data.get('vocal_features', {})
        hs = data.get('hume_probs', {})
        row = {'entry_id': entry_id, 'sentiment': sentiment}
        for k, v in vf.items():
            if isinstance(v, dict):
                for sub, val in v.items(): row[f'{k}_{sub}'] = val
            else:
                row[k] = v
        for emo in EMO_LABELS:
            row[f'hume_{emo}'] = hs.get(emo, np.nan)
        records.append(row)
    return pd.DataFrame(records).set_index('entry_id')


# ---------------------------------------------
# Plotting & Analysis
# ---------------------------------------------
def plot_heatmap(df, row_feats, col_feats, title, sentiment='all'):
    sub  = df[row_feats + col_feats]
    corr = sub.corr().loc[row_feats, col_feats]

    # Plot
    fig, ax = plt.subplots(figsize=(len(col_feats), len(row_feats)))
    sns.heatmap(corr, annot=True, center=0, cmap='RdBu_r', fmt='.2f', ax=ax)
    ax.set_title(f"{title} ({sentiment})")
    plt.tight_layout()

    # Save figure
    filename = f"{title.lower().replace(' ','_')}_{sentiment}.png"
    path = os.path.join(PLOT_DIR, filename)
    plot_and_save(fig, path)

    # Export to Excel
    excel_fn = os.path.join(EXPORT_DIR,
        f"{title.lower().replace(' ','_')}_{sentiment}.xlsx"
    )
    corr.to_excel(excel_fn)
    print(f"Saved correlation matrix to Excel: {excel_fn}")


"""
Uses plot_heatmap 
"""
def run_clip_level_plots(df, sentiment):

    fv = ACOUSTIC_FEATURES
    hv = [f'hume_{e}'   for e in LABELS]
    pv = [f'praat_{e}'  for e in LABELS]
    cv = [f'custom_{e}'    for e in LABELS]

    #hume
    plot_heatmap(df, fv, hv, "Vocal vs Hume correlations", sentiment)
   #praat
    df2 = df.rename(columns={p: c for p, c in zip(pv, cv)})
    plot_heatmap(df2, fv, cv, "Vocal vs Custom Categorization correlations", sentiment)

def summarize_anova_results(df, sentiment):
    fv = ACOUSTIC_FEATURES
    summary=[]
    if not df.empty:
        df['hume_label']=df[[f'hume_{e}' for e in LABELS]].idxmax(axis=1).str.replace('hume_','')
        for feat in fv:
            model=ols(f"{feat}~C(hume_label)",data=df).fit()
            anova=sm.stats.anova_lm(model,typ=2)
            p=anova['PR(>F)'].iloc[0]
            summary.append({'Feature':feat,'ANOVA p-value':round(p,4),'Significant?':p<0.05})
    df_sum=pd.DataFrame(summary)
    print(f"ANOVA Summary ({sentiment}):")
    print(df_sum)
    df_sum.to_excel(os.path.join(EXPORT_DIR,f"anova_summary_{sentiment}.xlsx"),index=False)


# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__=='__main__':
    df_clips_all=build_clip_level_df()
    df_segs_all=build_segment_df()
    df_full_all=build_clip_full_df()
    run_clip_level_plots(df_clips_all, sentiment = "all")
    for sentiment in SENTIMENTS:
        print(f"\n===== {sentiment.upper()} =====")
        #dfc=filter_by_sentiment(df_clips_all,sentiment)
        #run_clip_level_plots(dfc, sentiment)
        #summarize_anova_results(dfc, sentiment)