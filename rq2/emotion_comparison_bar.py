import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import audio_files, active_audio_id

with open("results_combined.json", "r") as f:
    results = json.load(f)
entry_id = active_audio_id
audio_path = audio_files[entry_id]["m4a"]

clip = results.get(entry_id)


hume_emotions = clip["hume_emotions"]  
nlp_emotions = clip["nlp_emotions"]    

hume_emotions = {k.lower(): v for k, v in hume_emotions.items() if k.lower() != "time"}
nlp_emotions = {k.lower(): v for k, v in nlp_emotions.items()}

emotion_categories = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

hume_scores = [hume_emotions.get(emo, np.nan) for emo in emotion_categories]
nlp_scores = [nlp_emotions.get(emo, np.nan) for emo in emotion_categories]

x = np.arange(len(emotion_categories))  # positions of the groups
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(10,6))
bars1 = ax.bar(x - width/2, hume_scores, width, label='Hume (Speech-Based)', color='red')
bars2 = ax.bar(x + width/2, nlp_scores, width, label='NLP Cloud (Text-Based)', color='blue')

ax.set_xlabel("Emotion Categories")
ax.set_ylabel("Score")
ax.set_title("Comparison of Emotion Scores: Speech vs. Text")
ax.set_xticks(x)
ax.set_xticklabels(emotion_categories, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()
