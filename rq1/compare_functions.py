## FOR CATEGORISATION FUNCTION 
## DIFFERENT VERSIONS INCLUDED HERE 

import json, re, sys, os
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    f1_score, recall_score, confusion_matrix, classification_report
)
import warnings, numpy as np
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze

EXPORT_DIR = Path("exports_sista")          
EXPORT_DIR.mkdir(exist_ok=True)

COMP_DIRS = {                         
    "SdZ-Scores" : Path("comparisons_rq1_old"),
    "V0_globalK" : Path("new_rq1_run_V0"),
    "V1_perK"    : Path("new_rq1_run_V1"),
    "V2_anchor"  : Path("new_rq1_run_V2"),
    "V3_bonus"   : Path("new_rq1_run_V3"),   
    "V4_bonSad" : Path("new_rq1_run_V4")
}


PARAM_RANGES = {
    "K_EXTREME"       : "0.6–1.6",
    "K_NEAR"          : "1.0–1.5",
    "bonus_Joy"       : "0 / 1",
    "min_anchor"      : "0 / 1",
}

VALUES_USED = {
    "SdZ-Scores": dict(K_EXTREME="-", fallback_e="-", loudness="1.0", min_anchor="-"),
    "V0_globalK": dict(K_EXTREME="1.6", fallback_e="-", loudness="1.0", min_anchor="-"),
    "V1_perK"   : dict(K_EXTREME="{joy0.5 anger1.6}",  fallback_e="0.25", loudness="0.6", min_anchor="-"),
    "V2_anchor" : dict(K_EXTREME="{joy0.5 anger1.2}",  fallback_e="0.25", loudness="0.6", min_anchor="2"),
    "V3_bonus"  : dict(K_EXTREME="{joy0.7 anger1.4}",  fallback_e="0.25", loudness="0.6", min_anchor="2"),
    "V4_bonSad"  : dict(K_EXTREME="{joy0.7 anger1.4}",  fallback_e="0.25", loudness="0.6", min_anchor="2"),

}

LABELS = ["anger", "joy", "sadness", "fear", "surprise"]
RX_SENT = re.compile(r"(?:^|[_\-\b])(pos|neg)(?:[_\-\b]|$)", re.I)

def sentiment_of(path: Path)->str:
    m = RX_SENT.search(path.stem)
    return m.group(1).lower() if m else "all"


def load_clips(folder: Path):
    Y, P, S = [], [], []           
    for fp in folder.glob("*.json"):
        with fp.open() as f:
            d = json.load(f)
        Y.append(d["hume_label"])
        P.append(d["praat_label"])
        S.append(sentiment_of(fp))
    return Y, P, S


abl_rows, distr = [], {sp: [] for sp in ("all","pos","neg")}
cm_sheets = {}  

for var, folder in COMP_DIRS.items():
    y_true, y_pred, sent = load_clips(folder)
    #metrics on ALL clips
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    uar   = recall_score(y_true, y_pred, labels=LABELS, average="macro")
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    abl_rows.append({"Variant":var, "Macro-F1":macro, "UAR":uar,
                     "Joy Rec":recalls[1], "Anger Rec":recalls[0],
                     "Surp FP":sum((p=="surprise" and t!="surprise") for p,t in zip(y_pred,y_true))})


    cm_sheets[var] = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                                  index=LABELS, columns=LABELS)
    

    for split in ("all","pos","neg"):
        mask = [True if split=="all" else s==split for s in sent]
        sub_preds = [p for p,m in zip(y_pred,mask) if m]
        counts = {lab: sub_preds.count(lab) for lab in LABELS}
        distr[split].append({"Variant":var, "N": len(sub_preds), **counts})


param_rows = []
for var in COMP_DIRS:
    row = {"Variant": var}
    for p in PARAM_RANGES:
        row[p] = VALUES_USED[var].get(p,"—")
    param_rows.append(row)


xlsf = EXPORT_DIR / "NY_thesis_tables-105.xlsx"
with pd.ExcelWriter(xlsf, engine="xlsxwriter") as xl:

    pd.DataFrame(abl_rows).round(3).to_excel(xl, sheet_name="ablation", index=False)

    for sp in distr:
        pd.DataFrame(distr[sp]).to_excel(xl, sheet_name=f"labels_{sp}", index=False)
   
    pd.DataFrame(param_rows).to_excel(xl, sheet_name="param_ledger", index=False)
  
    for var, df in cm_sheets.items():
        df.to_excel(xl, sheet_name=f"CM_{var}")

print(f"OK: Tables saved at {xlsf}")