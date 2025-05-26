"""
    Compute Pearson r and p-value for each emotion between hume_prob and praat_score.
    Returns a DataFrame with columns ['Emotion', "Pearson's r", 'p-value', 'Significant']
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from config_rq1 import INPUT_DIR_V3, EMO_LABELS
from common_functions import (load_sentiment_records, filter_by_sentiment, SENTIMENTS)
LABELS = ["anger","joy","sadness", "fear","surprise"]

INPUT_DIR = INPUT_DIR_V3
## Load all clips 
def load_all(comp_dir: str) -> pd.DataFrame:

    df = load_sentiment_records(
        dirpath=comp_dir,
        labels=LABELS,
        praat_key="praat_scores",
        hume_key="hume_probs",
        ext=".json"
    )

    df = df.rename(columns={
        'filename': 'clip',
        'custom_score': 'praat_score',
        'hume_score': 'hume_prob'
    })
    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    
    out = []
    for emo in EMO_LABELS:
        sub = df[df.emotion == emo].dropna(subset=['hume_prob', 'praat_score'])
        if len(sub) >= 2:
            r, p = pearsonr(sub.hume_prob, sub.praat_score)
        else:
            r, p = np.nan, np.nan
        out.append({
            'Emotion': emo.title(),
            "Pearson's r": round(r, 3),
            'p-value': round(p, 4),
            'Significant': 'Yes' if (not np.isnan(p) and p < 0.05) else 'No'
        })
    return pd.DataFrame(out)

def main():
    df_all = load_all(INPUT_DIR)
    for sentiment in SENTIMENTS:
        print(f"=== {sentiment.title()} Recordings ===")
        df_sub = filter_by_sentiment(df_all, sentiment)
        print(compute_correlations(df_sub).to_string(index=False))

if __name__ == "__main__":
    main()
