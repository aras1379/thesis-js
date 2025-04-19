import os
import json
from praat_parselmouth.vocal_extract import extract_features
from hume_ai.hume_utils import load_hume_average
from utils.compare_vectors import one_hot_vector, hume_vector, cosine_similarity
from hume_ai.hume_utils import normalize_emotions, combine_surprise_scores
from utils.categorize_vocal_emotions import categorize_emotion_from_vocal_markers  # your function from before
from config import audio_files

results = []

for entry_id, paths in audio_files.items():
    audio_path = paths["wav"]
    hume_path = f"hume_ai/filtered_results/average_raw/{entry_id}_average_raw_emotions.json"
    
    try:
        # 1. Extract vocal features and label
        vocal_features = extract_features(audio_path)
        praat_label = categorize_emotion_from_vocal_markers(vocal_features)
        praat_vec = one_hot_vector(praat_label)

        # 2. Load and normalize Hume emotions
        hume_raw = load_hume_average(hume_path)
        hume_scores = normalize_emotions(hume_raw)
        hume_vec = hume_vector(hume_scores)

        # 3. Compute cosine similarity
        sim = cosine_similarity(praat_vec, hume_vec)

        results.append({
            "entry_id": entry_id,
            "praat_label": praat_label,
            "cosine_similarity": round(sim, 4)
        })
    
    except Exception as e:
        print(f"Error processing {entry_id}: {e}")
        continue

# Print results
import pandas as pd
df = pd.DataFrame(results)
print(df)

# Optional: export or visualize
df.to_csv("exports/praat_vs_hume_cosine.csv", index=False)
