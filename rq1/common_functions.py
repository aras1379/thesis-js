"""
File with functions used generally 
Filter sentiments 
Load dataframes 
"""

import os, json
import numpy as np
import pandas as pd

SENTIMENTS = ("all", "positive", "negative")

def infer_sentiment(fn: str) -> str:
    """
    From any filename, return one of "positive", "negative", or "all".
    """
    lower = os.path.basename(fn).lower()
    if "_pos_" in lower:
        return "positive"
    if "_neg_" in lower:
        return "negative"
    return "all"

def group_by_sentiment_files(
    dirpath: str,
    ext: str = ".json"
) -> dict[str, list[str]]:
    """
    Scan dirpath for files ending in `ext` and group them by inferred sentiment.
    Returns a dict with keys "all", "positive", "negative".
    """
    groups = {s: [] for s in SENTIMENTS}
    for fn in sorted(os.listdir(dirpath)):
        if not fn.lower().endswith(ext):
            continue
        full = os.path.join(dirpath, fn)
        sent = infer_sentiment(fn)
        groups[sent].append(full)

        groups["all"].append(full)
    return groups


def filter_by_sentiment(
    df: pd.DataFrame,
    sentiment: str = "all"
) -> pd.DataFrame:
    """
    Slice a DataFrame that has a 'sentiment' column.
    If sentiment == 'all' returns df.copy(), else returns df[df.sentiment==sentiment].
    Prints counts of clips and rows for non-'all'.
    """
    if sentiment == "all":
        return df.copy()
    out = df[df["sentiment"] == sentiment].copy()
    # determine identifier column
    if "clip" in out.columns:
        id_col = "clip"
    elif "entry_id" in out.columns:
        id_col = "entry_id"
    else:
        id_col = None

    if id_col:
        n_clips = out[id_col].nunique()
    else:
        n_clips = out.index.nunique()
    n_rows = len(out)

    print(f"{n_clips} clips ({n_rows} rows) for '{sentiment}'")
    return out


"""
Load data from all files 
"""
def load_sentiment_records(
    dirpath: str,
    labels: list[str],
    praat_key: str = "praat_scores",
    hume_key: str = "hume_probs",
    ext: str = ".json"
) -> pd.DataFrame:
    """
      - praat_score  = data[praat_key].get(label, np.nan)
      - hume_score   = data[hume_key].get(label, np.nan)
    Returns a DataFrame with columns:
      ["filename", "sentiment", "emotion", "praat_score", "hume_score"]
    """
    records = []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.lower().endswith(ext):
            continue
        full = os.path.join(dirpath, fn)
        sent = infer_sentiment(fn)
        data = json.load(open(full, encoding="utf-8"))
        praat = data.get(praat_key, {})
        hume  = data.get(hume_key, {})
        for emo in labels:
            records.append({
                "filename":    fn,
                "sentiment":   sent,
                "emotion":     emo,
                "custom_score": float(praat.get(emo, np.nan)),
                "hume_score":  float(hume.get(emo, np.nan)),
            })
    return pd.DataFrame.from_records(records)