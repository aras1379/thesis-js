import json
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files  
from utils.data_utils import plot_and_save
EMOS = ["anger", "joy", "sadness", "fear", "surprise"]
RESULTS_FILE = "results_combined_normalized_percent.json"
EXPORT_DIR = "exports_rq2"
PLOT_DIR = "plots_rq2"

os.makedirs(EXPORT_DIR, exist_ok=True)

import json
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files
from utils.data_utils import plot_and_save

EMOS = ["anger", "joy", "sadness", "fear", "surprise"]
RESULTS_FILE = "results_combined_normalized_percent.json"
EXPORT_DIR = "exports_rq2"
PLOT_DIR = "plots_rq2"

os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------------------------------------
# Data Loading
# ---------------------------------------------
def load_and_prepare_data():
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for full_id, rec in data.items():
        parts = full_id.split("_")
        base_id = parts[1]
        sentiment = "positive" if parts[2] == "pos" else "negative"
        audio_path = audio_files.get(base_id, {}).get("m4a", "")

        h_raw = np.array([rec["hume_emotions"].get(e, 0.0) for e in EMOS])
        n_raw = np.array([rec["nlp_emotions"].get(e, 0.0) for e in EMOS])
        h_norm = h_raw / h_raw.sum() if h_raw.sum() > 0 else h_raw
        n_norm = n_raw / n_raw.sum() if n_raw.sum() > 0 else n_raw

        row = {
            "full_id": full_id,
            "entry_id": base_id,
            "sentiment": sentiment,
            "audio_path": audio_path
        }
        for i, e in enumerate(EMOS):
            row[f"hume_{e}"] = h_norm[i]
            row[f"nlp_{e}"] = n_norm[i]
        rows.append(row)

    return pd.DataFrame(rows).set_index("full_id")

# ---------------------------------------------
# Correlation Table
# ---------------------------------------------
def generate_correlation_table(df, sentiment="all"):
    records = []
    for e in EMOS:
        r, p = (np.nan, np.nan) if len(df) <= 1 else pearsonr(df[f"hume_{e}"], df[f"nlp_{e}"])
        records.append({
            "Emotion": e.title(),
            "Pearson_r": round(r, 3),
            "p-value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })
    corr_df = pd.DataFrame(records).set_index("Emotion")
    fn = f"correlations_{sentiment.lower()}.xlsx"
    corr_df.to_excel(os.path.join(EXPORT_DIR, fn))
    print(f"Saved correlation table: {fn}")

# ---------------------------------------------
# Difference Bar Plot
# ---------------------------------------------
def plot_difference_bars(df, sentiment="all"):
    label = "All clips" if sentiment.lower() == "all" else sentiment.capitalize()
    diffs = [(df[f"hume_{e}"] - df[f"nlp_{e}"]).mean() for e in EMOS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("pastel", n_colors=len(EMOS))
    ax.bar(EMOS, diffs, color=colors)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel("Mean(Hume - NLP)")
    ax.set_title(f"Average Difference: Hume vs NLP per Emotion ({label})")
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, f"hume_nlp_difference_{sentiment.lower()}")
    plot_and_save(fig, out)
    print(f"Saved difference bar plot: {out}.pdf")

# ---------------------------------------------
# Summary Table
# ---------------------------------------------
def generate_summary_table(df, sentiment="all"):
    summary = []
    for e in EMOS:
        h = df[f"hume_{e}"]
        n = df[f"nlp_{e}"]
        summary.append({
            "Emotion": e.title(),
            "Hume Mean": round(h.mean(), 3),
            "NLP Mean": round(n.mean(), 3),
            "Hume Std": round(h.std(), 3),
            "NLP Std": round(n.std(), 3),
            "Max Diff": round((h - n).abs().max(), 3),
            "Dominant": "Hume" if h.mean() > n.mean() else "NLP"
        })
    df_sum = pd.DataFrame(summary).set_index("Emotion")
    filename = f"hume_nlp_summary_{sentiment.lower()}.xlsx"
    df_sum.to_excel(os.path.join(EXPORT_DIR, filename))
    print(f"Saved summary table: {filename}")

# ---------------------------------------------
# Sentiment Comparison Plots
# ---------------------------------------------
def compare_sentiments(df, sentiment="all"):
    df = df.copy()
    df["Sentiment"] = df["sentiment"].str.capitalize()
    if sentiment == "all":
        targets = ["Positive", "Negative"]
    else:
        targets = [sentiment.capitalize()]

    records = []
    for sys in ["hume", "nlp"]:
        for targ in targets:
            mean_scores = df[df["Sentiment"] == targ][[f"{sys}_{e}" for e in EMOS]].mean()
            for e in EMOS:
                records.append({
                    "System": sys.upper(),
                    "Sentiment": targ,
                    "Emotion": e.title(),
                    "Mean": mean_scores[f"{sys}_{e}"]
                })
    sum_df = pd.DataFrame(records)
    # save table
    tbl = f"sentiment_comparison_summary_{sentiment}.xlsx"
    sum_df.to_excel(os.path.join(EXPORT_DIR, tbl), index=False)
    print(f"Saved summary table: {tbl}")

    # all-clips grouped bar
    if sentiment == "all":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=sum_df, x="Emotion", y="Mean", hue="System",
            errorbar=None, palette="Set2", ax=ax
        )
        ax.set(title="Hume vs NLP Mean Scores (All)", ylabel="Mean Score")
        plt.tight_layout()
        plot_and_save(fig, os.path.join(PLOT_DIR, "sentiment_comparison_all"))

        # faceted
        g = sns.catplot(
            data=sum_df, kind="bar",
            x="Emotion", y="Mean", hue="System", col="Sentiment",
            errorbar=None, palette="Set2",  height=6, aspect=1
        )
        g.set_axis_labels("Emotion", "Mean Score")
        g.set_titles("Sentiment: {col_name}")
        plt.tight_layout()
        g.savefig(os.path.join(PLOT_DIR, "sentiment_comparison_facet_all.pdf"))
        print("Saved faceted comparison plot.")
    else:
        g = sns.catplot(
            data=sum_df, kind="bar",
            x="Emotion", y="Mean", hue="System", col="Sentiment",
            errorbar=None, palette="Set2", height=6, aspect=1
        )
        g.set_axis_labels("Emotion", "Mean Score")
        g.set_titles("Sentiment: {col_name}")
        plt.tight_layout()
        g.savefig(os.path.join(PLOT_DIR, f"sentiment_comparison_{sentiment}.pdf"))
        print(f"Saved {sentiment} comparison plot.")

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    df = load_and_prepare_data()
    generate_correlation_table(df)
    plot_difference_bars(df, "all")
    # generate_summary_table(df)
    # compare_sentiments(df, "all")
    # compare_sentiments(df, "negative")
    # compare_sentiments(df, "positive")
    
    # for sentiment in ["negative", "positive"]:
    #     df_sent = filter_by_sentiment(df, sentiment)
    #     print(df_sent.index.tolist())
    #     generate_correlation_table(df_sent, sentiment)
    #     plot_correlation_heatmap(df_sent, sentiment)
    #     plot_difference_bars(df_sent, sentiment)
    #     generate_summary_table(df_sent, sentiment)
    #     compute_ttests_and_effect_size(df_sent, sentiment) 


if __name__ == "__main__":
    main()