import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files
from utils.data_utils import plot_and_save

# ==== GLOBAL CONFIG ====
RESULTS_FILE = "results_combined_rq2_rq3.json"
EXPORT_DIR   = "exports_rq3"
PLOT_DIR     = "plots_rq3"
EMOS         = ["anger", "joy", "sadness", "fear", "surprise"]

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==== LOAD DATA ====
def load_data():
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    rows = []
    for entry, rec in data.items():
        row = {"entry_id": entry}
        for e in EMOS:
            row[f"hume_{e}"] = rec["hume_emotions"].get(e, np.nan)
            row[f"nlp_{e}"]  = rec["nlp_emotions"].get(e, np.nan)
            row[f"self_{e}"] = rec["self_assessed"].get(e, np.nan)
        path = audio_files.get(entry, {}).get("m4a", "")
        if "positive" in path:
            row["Sentiment"] = "Positive"
        elif "negative" in path:
            row["Sentiment"] = "Negative"
        else:
            row["Sentiment"] = "Unknown"
        rows.append(row)

    return pd.DataFrame(rows).set_index("entry_id")

# ==== CORRELATION ANALYSIS ====
def compute_and_plot_correlations(df, sentiment="all"):
    # Filter by sentiment
    if sentiment != "all":
        df = df[df["Sentiment"] == sentiment.capitalize()]

    #  Pearson r, p-value, and significance
    hs, ns = [], []
    for e in EMOS:
        r1, p1 = pearsonr(df[f"hume_{e}"], df[f"self_{e}"])
        r2, p2 = pearsonr(df[f"nlp_{e}"], df[f"self_{e}"])
        hs.append({
            "emotion":    e,
            "pearson_r":  r1,
            "p_value":    p1,
            "Significant": "Yes" if p1 < 0.05 else "No"
        })
        ns.append({
            "emotion":    e,
            "pearson_r":  r2,
            "p_value":    p2,
            "Significant": "Yes" if p2 < 0.05 else "No"
        })

    hs_df = pd.DataFrame(hs).set_index("emotion")
    ns_df = pd.DataFrame(ns).set_index("emotion")

    # Save heatmaps + tables
    for title, df_corr in [("hume_vs_self", hs_df), ("nlp_vs_self", ns_df)]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        sns.heatmap(
            df_corr[["pearson_r"]],
            annot=True, cmap="coolwarm", center=0,
            fmt=".2f", cbar_kws={"label": "Pearson r"},
            ax=axes[0]
        )
        axes[0].set_title(f"{title.replace('_',' ').title()} r ({sentiment})")
        axes[0].set_ylabel("Emotion")

        sns.heatmap(
            df_corr[["p_value"]],
            annot=True, cmap="viridis_r", center=0.05,
            fmt=".3f", cbar_kws={"label": "p-value"},
            ax=axes[1]
        )
        axes[1].set_title(f"{title.replace('_',' ').title()} p ({sentiment})")
        axes[1].set_ylabel("")

        plt.tight_layout()
        fname = f"heatmap_{title}_{sentiment.lower()}"
        plot_and_save(fig, os.path.join(PLOT_DIR, fname))

        # include the new Significant column when saving
        out_fn = os.path.join(EXPORT_DIR, f"corr_{title}_{sentiment.lower()}.xlsx")
        df_corr.to_excel(out_fn)
        print(f"Saved correlation table with significance: {out_fn}")

    return hs_df, ns_df


# ==== SCATTER PLOTS ====
def plot_scatter_per_emotion(df, hs_df, ns_df, sentiment="all"):
    if sentiment != "all":
        df = df[df["Sentiment"] == sentiment.capitalize()]

    for e in EMOS:
        fig, axs = plt.subplots(1,2, figsize=(8,4), sharey=True)
        for ax, (method, df_corr) in zip(axs, [("hume", hs_df), ("nlp", ns_df)]):
            x = df[f"{method}_{e}"]
            y = df[f"self_{e}"]
            r = df_corr.loc[e, "pearson_r"]
            p = df_corr.loc[e, "p_value"]
            sns.regplot(x=x, y=y, ci=95, scatter_kws={"alpha":0.6}, ax=ax)
            ax.set_title(f"{method.upper()} vs Self ({sentiment})\n"
                         f"r={r:.2f}, p={p:.3f}")
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.set_xlabel(f"{method.upper()} score")
            if method=="hume":
                ax.set_ylabel("Self-assess")

        plt.suptitle(e.title())
        plt.tight_layout(rect=(0,0,1,0.95))
        fname = f"scatter_{e}_vs_self_{sentiment.lower()}"
        plot_and_save(fig, os.path.join(PLOT_DIR, fname))

# ==== DESCRIPTIVE SUMMARY TABLE ====
def generate_rq3_summary_table(df, sentiment="all"):
    if sentiment != "all":
        df = df[df["Sentiment"] == sentiment.capitalize()]

    summary = []
    for e in EMOS:
        summary.append({
            "Emotion":            e.title(),
            "Self-Reported Mean": round(df[f"self_{e}"].mean(), 3),
            "Hume Mean":          round(df[f"hume_{e}"].mean(), 3),
            "NLP Mean":           round(df[f"nlp_{e}"].mean(), 3),
            "Self-Reported Std":  round(df[f"self_{e}"].std(), 3),
            "Hume Std":           round(df[f"hume_{e}"].std(), 3),
            "NLP Std":            round(df[f"nlp_{e}"].std(), 3)
        })

    summary_df = pd.DataFrame(summary).set_index("Emotion")
    print(f"\nRQ3 Summary Table ({sentiment}):")
    print(summary_df)

    fn = os.path.join(EXPORT_DIR, f"rq3_emotion_summary_{sentiment.lower()}.xlsx")
    summary_df.to_excel(fn)
    print(f"Saved summary table: {fn}")
    return summary_df

# ==== PLOT BAR CHART ====
def plot_rq3_bar_chart(summary_df, sentiment="all"):
    # summary_df already built for correct sentiment
    plot_df = summary_df[["Hume Mean","NLP Mean","Self-Reported Mean"]].reset_index()
    plot_df = pd.melt(plot_df, id_vars="Emotion", var_name="Source", value_name="Mean Score")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=plot_df, x="Emotion", y="Mean Score", hue="Source", palette="Set2", ax=ax)
    ax.set_title(f"Average Emotion Scores (RQ3: {sentiment})")
    ax.set_ylabel("Mean Score")
    plt.tight_layout()

    fname = f"rq3_emotion_comparison_{sentiment.lower()}"
    plot_and_save(fig, os.path.join(PLOT_DIR, fname))

# ==== SENTIMENT-BASED BAR CHART (already does both facets) ====
def plot_rq3_sentiment_bar(df):
    # unchanged
    summaries = []
    for sentiment in ['Positive','Negative']:
        subset = df[df['Sentiment']==sentiment]
        for source in ['hume','nlp','self']:
            means = subset[[f"{source}_{e}" for e in EMOS]].mean()
            for emo in EMOS:
                summaries.append({
                    'Sentiment': sentiment,
                    'Emotion': emo.title(),
                    'Source': source.upper(),
                    'Mean_Score': means[f"{source}_{emo}"]
                })

    summary_df = pd.DataFrame(summaries)
    g = sns.catplot(
        data=summary_df, kind="bar",
        x="Emotion", y="Mean_Score",
        hue="Source", col="Sentiment",
        palette="Set2", ci=None, height=6, aspect=1
    )
    g.set_axis_labels("Emotion","Mean Emotion Score")
    g.set_titles("Sentiment: {col_name}")
    g._legend.set_title("Source")
    plt.tight_layout()
    plot_and_save(g.fig, os.path.join(PLOT_DIR, "rq3_sentiment_grouped_bar"))

# ==== PAIRED T-TESTS vs SELF ====
def compute_ttests_vs_self(df, sentiment="all"):
    if sentiment != "all":
        df = df[df["Sentiment"] == sentiment.capitalize()]

    records = []
    for source in ["hume","nlp"]:
        for e in EMOS:
            a = df[f"{source}_{e}"]
            b = df[f"self_{e}"]
            if len(df)>1:
                t,p = ttest_rel(a,b, nan_policy="omit")
                d   = (a-b).mean()/ (a-b).std(ddof=1)
            else:
                t=p=d=np.nan
            records.append({
                "Sentiment":      sentiment.capitalize(),
                "Comparison":     f"{source.upper()} vs Self",
                "Emotion":        e.title(),
                "t-statistic":    round(t,3),
                "p-value":        round(p,4),
                "Significant?":   "Yes" if p<0.05 else "No",
                "Cohen's d":      round(d,3)
            })

    tdf = pd.DataFrame(records)
    fn = os.path.join(EXPORT_DIR, f"ttests_self_vs_ai_{sentiment.lower()}.xlsx")
    tdf.to_excel(fn, index=False)
    print(f"Saved self-vs-AI t-tests & d: {fn}")
    return tdf

# ==== MAIN ====
def main():
    df = load_data()

    # correlations + scatters for each sentiment
    # for sentiment in ["all"]:
    #     compute_and_plot_correlations(df, sentiment)
        #plot_scatter_per_emotion(df, hs_df, ns_df, sentiment)

    # summary table + bar chart
    # for sentiment in ["all"]:
    #     summ_df = generate_rq3_summary_table(df, sentiment)
    #     plot_rq3_bar_chart(summ_df, sentiment)


    plot_rq3_sentiment_bar(df)

    # t-tests vs self
    for sentiment in ["all","positive","negative"]:
        compute_ttests_vs_self(df, sentiment)

if __name__ == "__main__":
    main()
