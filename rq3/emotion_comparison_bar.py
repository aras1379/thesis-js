import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files, active_audio_id
from utils.data_utils import plot_and_save

# Load RQ2/RQ3 results
with open("results_combined_rq2_rq3.json", "r") as f:
    results = json.load(f)

entry_id = active_audio_id
clip = results[entry_id]

# 1) Extract scores
hume_raw = clip["hume_emotions"]
nlp_raw  = clip["nlp_emotions"]
self_raw = clip["self_assessed"]

EMOS = ["anger", "joy", "sadness", "fear", "surprise"]

hume_scores = [hume_raw.get(emo, np.nan) for emo in EMOS]
nlp_scores  = [nlp_raw.get(emo, np.nan)  for emo in EMOS]
self_scores = [self_raw.get(emo, np.nan) for emo in EMOS]

# 2) Bar plot
x = np.arange(len(EMOS))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w, hume_scores, w, label="Hume (Speech)", color="red")
ax.bar(x,     nlp_scores,  w, label="NLP (Text)",    color="blue")
ax.bar(x + w, self_scores, w, label="Self‑assessed", color="green")

ax.set_xticks(x)
ax.set_xticklabels([e.title() for e in EMOS], rotation=45)
ax.set_ylabel("Score")
ax.set_title(f"Emotion Scores: Speech vs Text vs Self, clip {entry_id}")
ax.legend()
plt.tight_layout()

# Save plot using plot_and_save

plot_filename = os.path.join("plots_rq3", f"speech_text_self_comparison_{entry_id}")
plot_and_save(fig, plot_filename)

# 3) Table + Excel
df = pd.DataFrame({
    "Emotion":       [e.title() for e in EMOS],
    "Hume (Speech)": hume_scores,
    "NLP (Text)":    nlp_scores,
    "Self‑Assessed": self_scores
})

print("\nComparison Table:")
print(df.to_string(index=False))


excel_out = os.path.join("exports_rq3", f"emotion_scores_comparison_{entry_id}.xlsx")
df.to_excel(excel_out, index=False)
print(f"\nSaved Excel file: {excel_out}")