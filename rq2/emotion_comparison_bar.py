import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files, active_audio_id
from utils.data_utils import plot_and_save

# Load data
with open("results_combined_normalized_percent.json", "r") as f:
    results = json.load(f)

entry_id = active_audio_id
audio_path = audio_files[entry_id]["m4a"]
clip = results.get(entry_id)

# Extract emotion data
hume_emotions = {k.lower(): v for k, v in clip["hume_emotions"].items() if k.lower() != "time"}
nlp_emotions = {k.lower(): v for k, v in clip["nlp_emotions"].items()}

emotion_categories = ["anger", "joy", "sadness", "fear", "surprise"]

hume_scores = [hume_emotions.get(emo, np.nan) for emo in emotion_categories]
nlp_scores = [nlp_emotions.get(emo, np.nan) for emo in emotion_categories]

# Bar Plot
x = np.arange(len(emotion_categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, hume_scores, width, label='Hume (Speech-Based)', color='red')
bars2 = ax.bar(x + width/2, nlp_scores, width, label='NLP Cloud (Text-Based)', color='blue')

ax.set_xlabel("Emotion Categories")
ax.set_ylabel("Score")
ax.set_title(f"Comparison of Emotion Scores: Speech vs. Text, clip {entry_id}")
ax.set_xticks(x)
ax.set_xticklabels(emotion_categories, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()
output_dir = "plots_rq2"
os.makedirs(output_dir, exist_ok=True)
plot_filename = os.path.join(output_dir, f"speech_vs_text_emotions_{entry_id}")
plot_and_save(fig, plot_filename)

print(f"Saved comparison plot: {plot_filename}.pdf")

df = pd.DataFrame({
    "Emotion": [e.title() for e in emotion_categories],
    "Hume (Speech-Based)": hume_scores,
    "NLP Cloud (Text-Based)": nlp_scores
})

print("\nComparison Table:")
print(df.to_string(index=False))

# Save to Excel
output_dir = "exports_rq2"
os.makedirs(output_dir, exist_ok=True)
excel_filename = f"emotion_scores_comparison_{entry_id}.xlsx"
excel_path = os.path.join(output_dir, excel_filename)
df.to_excel(excel_path, index=False)
print(f"\nSaved Excel file: {excel_path}")
