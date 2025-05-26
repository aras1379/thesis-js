# USE 
# for FIG 4.3 rq1 4.2.3 

# Scatter plot of mean score rule-based vs hume
import os, sys 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import plot_and_save
from common_functions import load_sentiment_records, filter_by_sentiment, SENTIMENTS
from config_rq1 import PLOT_DIR, EMO_LABELS, EXPORT_DIR, INPUT_DIR_V3

EMOS = EMO_LABELS
def load_comparison_df(dirpath: str) -> pd.DataFrame:

    df = load_sentiment_records(
        dirpath=dirpath,
        labels=EMO_LABELS,
        praat_key='praat_scores',
        hume_key='hume_probs',
        ext='_vocal_vs_hume.json'
    )
    # rename praat 
    df = df.rename(columns={
        'filename':    'entry_id',
        'custom_score':'praat_score'
    })
    return df

# -----------------------------------------------------------------------------
# Plotting
# Rule-based vs hume scores each emotion average 
# -----------------------------------------------------------------------------
def build_comparison_scatter(
    df: pd.DataFrame,
    save_dir: str,
    sentiment: str
):

    emotions = EMO_LABELS
    offset   = 0.2
    x_pos    = np.arange(len(emotions))

    # compute means
    praat_means = df.groupby('emotion')['praat_score'].mean().reindex(emotions)
    hume_means  = df.groupby('emotion')['hume_score'].mean().reindex(emotions)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x_pos - offset/2, praat_means, marker='o', linestyle='--', label='Rule-based Mean')
    ax.plot(x_pos + offset/2, hume_means,  marker='s', linestyle='--', label='Hume Mean')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([e.title() for e in emotions], rotation=45)
    ax.set_ylabel('Score')
    ax.set_title(f'Rule-based vs Hume Emotion Scores ({sentiment.title()})')
    ax.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'praat_hume_{sentiment.lower()}_scatter')
    plot_and_save(fig, out_path)
    print(f"Saved plot: {out_path}.pdf")



# ----------------------------------
# main
def main():
    df_all = load_comparison_df(INPUT_DIR_V3)

    build_comparison_scatter(df_all, save_dir=PLOT_DIR, sentiment='all')

    for sentiment in ['negative', 'positive']:
        df_sub = filter_by_sentiment(df_all, sentiment)
        build_comparison_scatter(df_sub, save_dir=PLOT_DIR, sentiment=sentiment)

if __name__ == "__main__":
    main()