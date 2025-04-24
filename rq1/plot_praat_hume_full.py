import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Define your emotions of interest here:
emotions_to_analyze = ["anger","fear","joy","sadness","surprise"]

def build_comparison_table(comparisons_dir="comparisons_rq1"):
    """
    Reads all JSON files in comparisons_dir, extracts Praat scores and Hume probabilities
    for each emotion, and returns a DataFrame where each row is a clip.
    """
    records = []
    for fn in sorted(os.listdir(comparisons_dir)):
        if not fn.endswith("_vocal_vs_hume.json"):
            continue
        path = os.path.join(comparisons_dir, fn)
        with open(path, 'r') as f:
            data = json.load(f)
        entry = data.get("entry_id", fn.replace("_vocal_vs_hume.json", ""))

        praat = data.get("praat_scores", {})
        hume  = {k: v for k, v in data.get("hume_probs", {}).items() 
                 if k in emotions_to_analyze}

        row = {"entry_id": entry}
        for emo in emotions_to_analyze:
            row[f"{emo}_praat"] = praat.get(emo, pd.NA)
            row[f"{emo}_hume"]  = hume.get(emo, pd.NA)
        records.append(row)

    df = pd.DataFrame(records).set_index("entry_id")
    return df

def build_comparison_diagram():
    # Configuration
    comparisons_dir = "comparisons_rq1"
    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    width = 0.2  # horizontal offset for scatter

    # Load all data into lists
    records = []
    for fn in sorted(os.listdir(comparisons_dir)):
        if not fn.endswith("_vocal_vs_hume.json"):
            continue
        data = json.load(open(os.path.join(comparisons_dir, fn)))
        praat = data.get("praat_scores", {})
        hume  = data.get("hume_probs", {})
        for emo in emotions:
            records.append({
                "entry_id": fn.replace("_vocal_vs_hume.json", ""),
                "emotion": emo,
                "praat_score": praat.get(emo, np.nan),
                "hume_score":  hume.get(emo,  np.nan)
            })

    # Organize by emotion
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(emotions))

    for i, emo in enumerate(emotions):
        # extract scores for this emotion across clips
        praat_vals = [r["praat_score"] for r in records if r["emotion"] == emo]
        hume_vals  = [r["hume_score"]  for r in records if r["emotion"] == emo]
        # scatter plot with slight x-offset
        ax.scatter(
            np.full(len(praat_vals), x_positions[i] - width/2),
            praat_vals,
            label="Praat (Audio)" if i == 0 else "",
            marker="o",
            alpha=0.7
        )
        ax.scatter(
            np.full(len(hume_vals), x_positions[i] + width/2),
            hume_vals,
            label="Hume (Speech AI)" if i == 0 else "",
            marker="s",
            alpha=0.7
        )

        # Mean lines
        praat_means = [np.nanmean([r["praat_score"] for r in records if r["emotion"] == emo]) for emo in emotions]
        hume_means  = [np.nanmean([r["hume_score"]  for r in records if r["emotion"] == emo]) for emo in emotions]
        ax.plot(x_positions - width/2, praat_means, linestyle="--", marker="o", label="Praat Mean")
        ax.plot(x_positions + width/2, hume_means,  linestyle="--", marker="s", label="Hume Mean")

        # Formatting
        ax.set_xticks(x_positions)
        ax.set_xticklabels([e.title() for e in emotions], rotation=45)
        ax.set_ylabel("Score")
        ax.set_title("Praat vs. Hume Emotion Scores Across All Clips")
        ax.legend()
        plt.tight_layout()

        # Save and show
        output_dir = "exports"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "praat_hume_all_clips_scatter.png")
        fig.savefig(out_path)
        plt.show()

        print(f"Saved visual comparison: {out_path}")
        
def main():
    comp_dir = "comparisons_rq1"
    if not os.path.isdir(comp_dir):
        raise FileNotFoundError(f"No such directory: {comp_dir}")

    df = build_comparison_table(comp_dir)

    # 1) Print to console
    print("\n=== Praat vs Hume Comparison Table ===\n")
    print(df.to_string())
    
    build_comparison_diagram()

    # 2) Save to CSV
    os.makedirs("exports", exist_ok=True)
    csv_path = os.path.join("exports", "praat_hume_comparison.csv")
    df.to_csv(csv_path)
    print(f"\nSaved CSV: {csv_path}")

    # 3) (Optional) Save to Excel
    excel_path = os.path.join("exports", "praat_hume_comparison.xlsx")
    df.to_excel(excel_path)
    print(f"Saved Excel: {excel_path}")

if __name__ == "__main__":
    main()
