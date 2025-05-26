import os
import sys
import json
import numpy as np
import pandas as pd
import argparse
import parselmouth
from scipy.stats import pearsonr, ttest_ind

# Add project path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files
from praat_parselmouth.vocal_extract import extract_features
from common_functions import infer_sentiment, filter_by_sentiment, group_by_sentiment_files, SENTIMENTS
from utils.parselmouth_utils import segment_and_extract_features
# Directories & constants
COMPARISONS_DIR = "comparisons_rq1_2"
EXPORTS_DIR = "exports_rq1_new_2"
HUME_DIR = "hume_ai/filtered_results/filtered"
LABELS = ['anger','joy','sadness','fear','surprise']
ACOUSTIC_FEATURES = [
    'mean_pitch_hz','mean_intensity_db','mean_hnr_db',
    'jitter_local','shimmer_local'
]
SEGMENT_LENGTH = 1.25  # seconds

# Ensure output dir exists
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Load baseline for z-scoring
BASELINE_DF = pd.read_excel(os.path.join(EXPORTS_DIR, "vocal_features_baseline.xlsx"), index_col=0)

def analyze_single_clip(clip_id: str, wav_path: str, hume_dir: str, segment_length: float):
    # extract segments and z-score
    df_seg = segment_and_extract_features(wav_path, segment_length)
    # z-score
    for feat in ACOUSTIC_FEATURES:
        mu = BASELINE_DF.loc[feat, 'Baseline_Mean']
        sd = BASELINE_DF.loc[feat, 'Baseline_Std']
        df_seg[feat + '_z'] = (df_seg[feat] - mu) / sd
    # load Hume
    fn = next((f for f in os.listdir(hume_dir)
               if f.startswith(clip_id) and f.endswith('_filtered_emotions.json')), None)
    if not fn:
        raise FileNotFoundError(f"No Hume JSON for {clip_id}")
    hume = pd.DataFrame(json.load(open(os.path.join(hume_dir, fn))))
    hume.rename(columns={'time':'Segment_Start'}, inplace=True)
    # merge
    merged = pd.merge_asof(
        df_seg.sort_values('Segment_Start'),
        hume[['Segment_Start'] + LABELS].sort_values('Segment_Start'),
        on='Segment_Start', direction='nearest', tolerance=segment_length/2
    ).dropna(subset=[feat + '_z' for feat in ACOUSTIC_FEATURES] + LABELS)
    # correlations
    corr_rows = []
    t_rows    = []
    for feat in ACOUSTIC_FEATURES:
        zcol = feat + '_z'
        for emo in LABELS:
            sub = merged[[zcol, emo]].dropna()
            if len(sub) > 1:
                r,p = pearsonr(sub[zcol], sub[emo])
                thresh = sub[emo].quantile(0.7)
                high = sub[sub[emo]>=thresh][zcol]
                low  = sub[sub[emo]< thresh][zcol]
                t,p_t = ttest_ind(high, low, nan_policy='omit')
            else:
                r,p,p_t = np.nan, np.nan, np.nan
            corr_rows.append({'Clip': clip_id,'Feature':feat,'Emotion':emo,'Pearson_r':round(r,3),'p-value':round(p,4),'Significant':p<0.05 if not np.isnan(p) else False})
            t_rows.append({'Clip': clip_id,'Feature':feat,'Emotion':emo,'t-statistic':round(t,3),'p-value':round(p_t,4),'Significant':p_t<0.05 if not np.isnan(p_t) else False})
    corr_df = pd.DataFrame(corr_rows)
    ttest_df= pd.DataFrame(t_rows)
    # save
    corr_df.to_excel(os.path.join(EXPORTS_DIR, f'corr_clip_{clip_id}.xlsx'), index=False)
    ttest_df.to_excel(os.path.join(EXPORTS_DIR, f'ttest_clip_{clip_id}.xlsx'), index=False)
    print(f"Saved single-clip analyses for {clip_id}")
    return corr_df, ttest_df


def full_seg_level():
    # load all clips
    segments = []
    for clip_id, meta in audio_files.items():
        wav = meta.get('wav')
        if not wav or not os.path.exists(wav):
            continue
        df_seg = segment_and_extract_features(wav, SEGMENT_LENGTH)
        if df_seg.empty:
            continue
        # z-score
        for feat in ACOUSTIC_FEATURES:
            mu = BASELINE_DF.loc[feat, 'Baseline_Mean']
            sd = BASELINE_DF.loc[feat, 'Baseline_Std']
            df_seg[feat + '_z'] = (df_seg[feat] - mu) / sd
        # load Hume
        fn = next((f for f in os.listdir(HUME_DIR) if f.startswith(clip_id) and f.endswith('_filtered_emotions.json')), None)
        if not fn:
            continue
        hume = pd.DataFrame(json.load(open(os.path.join(HUME_DIR, fn))))
        hume.rename(columns={'time':'Segment_Start'}, inplace=True)
        merged = pd.merge_asof(
            df_seg.sort_values('Segment_Start'),
            hume[['Segment_Start'] + LABELS].sort_values('Segment_Start'),
            on='Segment_Start', direction='nearest', tolerance=SEGMENT_LENGTH/2
        )
        merged['clip_id'] = clip_id
        segments.append(merged.dropna(subset=[feat + '_z' for feat in ACOUSTIC_FEATURES]))
    if not segments:
        print("No segments to analyze.")
        return
    df_all = pd.concat(segments, ignore_index=True)
    # build sentiment sets
    sent_map = {s:set() for s in SENTIMENTS}
    for fn in os.listdir(COMPARISONS_DIR):
        if not fn.endswith('_vocal_vs_hume.json'): continue
        sid = json.load(open(os.path.join(COMPARISONS_DIR, fn))).get('entry_id')
        sent_map[infer_sentiment(fn)].add(sid)
        sent_map['all'].add(sid)
    # per-sentiment stats
    for sentiment in SENTIMENTS:
        df_s = df_all[df_all['clip_id'].isin(sent_map[sentiment])]
        if df_s.empty:
            print(f"No data for '{sentiment}'")
            continue
        # compute and save
        corr_rows,t_rows = [],[]
        for feat in ACOUSTIC_FEATURES:
            zcol = feat + '_z'
            for emo in LABELS:
                sub = df_s[[zcol,emo]].dropna()
                if len(sub)>1:
                    r,p = pearsonr(sub[zcol], sub[emo])
                    thresh = sub[emo].quantile(0.7)
                    high = sub[sub[emo]>=thresh][zcol]
                    low  = sub[sub[emo]< thresh][zcol]
                    t,p_t = ttest_ind(high, low, nan_policy='omit')
                else:
                    r,p,p_t = np.nan,np.nan,np.nan
                corr_rows.append({'Sentiment':sentiment,'Feature':feat,'Emotion':emo,'Pearson_r':round(r,3),'p-value':round(p,4),'Significant':p<0.05 if not np.isnan(p) else False})
                t_rows.append({'Sentiment':sentiment,'Feature':feat,'Emotion':emo,'t-statistic':round(t,3),'p-value':round(p_t,4),'Significant':p_t<0.05 if not np.isnan(p_t) else False})
        pd.DataFrame(corr_rows).to_excel(os.path.join(EXPORTS_DIR,f'corr_{sentiment}.xlsx'),index=False)
        pd.DataFrame(t_rows).to_excel(os.path.join(EXPORTS_DIR,f'ttest_{sentiment}.xlsx'),index=False)
        print(f"Saved analyses for sentiment '{sentiment}'")

def main():
    
    ## ONE CLIP: 
    # segment_analysis.py --clip id_012_neg 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', type=str, help="Clip ID to analyze (or omit for full dataset)")
    args = parser.parse_args()

    if args.clip:
        wav = audio_files.get(args.clip, {}).get('wav')
        if not wav:
            print(f"Unknown clip: {args.clip}")
            return
        analyze_single_clip(args.clip, wav, HUME_DIR, SEGMENT_LENGTH)
    else:
        full_seg_level()

if __name__ == '__main__':
    main()
