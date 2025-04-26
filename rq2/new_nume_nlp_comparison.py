import json
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files  
from utils.data_utils import plot_and_save
EMOS = ["anger", "joy", "sadness", "fear", "surprise"]
RESULTS_FILE = "results_combined_rq2_rq3.json"
EXPORT_DIR = "exports_rq2"
PLOT_DIR = "plots_rq2"

os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------------------------------------
def load_and_prepare_data():
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for entry, rec in data.items():
        h_raw = np.array([rec["hume_emotions"].get(e, 0.0) for e in EMOS])
        n_raw = np.array([rec["nlp_emotions"].get(e, 0.0) for e in EMOS])

        # Normalize
        h_norm = h_raw / h_raw.sum() if h_raw.sum() > 0 else h_raw
        n_norm = n_raw / n_raw.sum() if n_raw.sum() > 0 else n_raw

        row = {"entry_id": entry}
        for i, e in enumerate(EMOS):
            row[f"hume_{e}"] = h_norm[i]
            row[f"nlp_{e}"] = n_norm[i]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("entry_id")
    return df

# ---------------------------------------------
def plot_correlation_heatmap(df, sentiment="all"):
    corrs = []
    for e in EMOS:
        r, _ = pearsonr(df[f"hume_{e}"], df[f"nlp_{e}"])
        corrs.append(r)
    corr_df = pd.DataFrame({"Emotion": EMOS, "Pearson_r": corrs}).set_index("Emotion")

    plt.figure(figsize=(4,6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, cbar_kws={"label":"Pearson r"})
    plt.title(f"Hume vs NLP Correlation per Emotion ({sentiment})")
    plt.tight_layout()
    filename = f"hume_nlp_correlation_heatmap_{sentiment.lower()}.png"
    plt.savefig(os.path.join(EXPORT_DIR, filename))
    plt.show()

# ---------------------------------------------
def plot_difference_bars(df, sentiment="all"):
    diffs = []
    for e in EMOS:
        mean_diff = (df[f"hume_{e}"] - df[f"nlp_{e}"]).mean()
        diffs.append(mean_diff)
    
    # Skapa fig och ax
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=EMOS, y=diffs, palette="vlag", ax=ax)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel("Mean(Hume - NLP)")
    ax.set_title(f"Average Difference: Hume vs NLP per Emotion ({sentiment})")
    plt.tight_layout()

  
    filename = os.path.join(PLOT_DIR, f"hume_nlp_difference_{sentiment.lower()}")
    plot_and_save(fig, filename)

    print(f"Saved difference bar plot: {filename}.pdf")

# ---------------------------------------------
def generate_summary_table(df, sentiment="all"):
    summary = []
    for e in EMOS:
        h_mean = df[f"hume_{e}"].mean()
        n_mean = df[f"nlp_{e}"].mean()
        h_std  = df[f"hume_{e}"].std()
        n_std  = df[f"nlp_{e}"].std()
        max_diff = (df[f"hume_{e}"] - df[f"nlp_{e}"]).abs().max()
        dominant = "Hume" if h_mean > n_mean else "NLP"

        summary.append({
            "Emotion": e.title(),
            "Hume Mean": round(h_mean, 3),
            "NLP Mean": round(n_mean, 3),
            "Hume Std": round(h_std, 3),
            "NLP Std": round(n_std, 3),
            "Max Diff": round(max_diff, 3),
            "Dominant System": dominant
        })

    summary_df = pd.DataFrame(summary).set_index("Emotion")
    print(f"\nHume vs NLP Summary Table: ({sentiment})")
    print(summary_df)
    filename = f"hume_nlp_summery_{sentiment.lower()}.xlsx"

    summary_df.to_excel(os.path.join(EXPORT_DIR, filename))

# ---------------------------------------------

def filter_by_sentiment(df, sentiment="positive"):
    

    valid_ids = []
    for entry_id in df.index:
        audio_path = audio_files.get(entry_id, {}).get("m4a", "")
        if sentiment in audio_path:
            valid_ids.append(entry_id)
    
    filtered_df = df.loc[valid_ids]
    print(f"\nFiltered {len(filtered_df)} clips for sentiment: {sentiment.capitalize()}")
    return filtered_df

def compare_sentiments(df):
    from config import audio_files

    sentiment_map = {}
    for entry_id in df.index:
        path = audio_files.get(entry_id, {}).get("m4a", "")
        if "positive" in path:
            sentiment_map[entry_id] = "Positive"
        elif "negative" in path:
            sentiment_map[entry_id] = "Negative"

    df['Sentiment'] = df.index.map(sentiment_map)
    summaries = []

    for system in ['hume', 'nlp']:
        for sentiment in ['Positive', 'Negative']:
            subset = df[df['Sentiment'] == sentiment]
            means = subset[[f'{system}_{e}' for e in EMOS]].mean()
            for emo in EMOS:
                summaries.append({
                    'System': system.upper(),
                    'Sentiment': sentiment,
                    'Emotion': emo.title(),
                    'Mean_Score': means[f'{system}_{emo}']
                })

    summary_df = pd.DataFrame(summaries)

    # Plot using plot_and_save
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=summary_df, x='Emotion', y='Mean_Score',
        hue='System', palette='Set2', ci=None, hue_order=['HUME', 'NLP'], ax=ax
    )
    ax.set_title('Mean Emotion Scores by Sentiment (Positive vs Negative)')
    ax.set_ylabel('Mean Normalized Score')
    ax.legend(title='System')
    plt.tight_layout()

    # Save plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    filename = os.path.join(EXPORT_DIR, 'sentiment_comparison_bar')
    plot_and_save(fig, filename)

    # Export table
    summary_df.to_excel(os.path.join(EXPORT_DIR, 'sentiment_comparison_summary.xlsx'), index=False)
    print("Saved sentiment comparison analysis.")


def main():
    df = load_and_prepare_data()
    plot_correlation_heatmap(df)
    plot_difference_bars(df)
    generate_summary_table(df)
    
    compare_sentiments(df)
    
    df_neg = filter_by_sentiment(df, sentiment="negative")
    plot_correlation_heatmap(df_neg, sentiment="negative")
    plot_difference_bars(df_neg, sentiment="negative")


    generate_summary_table(df_neg, sentiment="negative")
    
    df_pos = filter_by_sentiment(df, sentiment="positive")
    plot_correlation_heatmap(df_pos, sentiment="positive")
    plot_difference_bars(df_pos, sentiment="positive")
    generate_summary_table(df_pos, sentiment="positive")

if __name__ == "__main__":
    main()