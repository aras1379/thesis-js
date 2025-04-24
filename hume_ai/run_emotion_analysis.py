#run_emotion_analysis.py

import time
import os
import json
import sys

from emotion_recognition import analyze_audio
from fetch_results import get_analysis_results
from average_functions import compute_average_emotions
from hume_utils import normalize_emotions, combine_surprise_scores

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import audio_files, active_audio_id

TARGET_EMOTIONS = {
    "Joy", "Sadness", "Fear", "Anger",
    "Surprise (positive)", "Surprise (negative)"
}

entry_id = active_audio_id
audio_path = audio_files[entry_id]["m4a"]

# Start the analysis
job_id = analyze_audio(audio_path)

# Wait for Hume to process the audio
print("Waiting for Hume AI job to finish...")
max_retries = 30
retry_delay = 5  

results = None
for attempt in range(max_retries):
    try:
        results = get_analysis_results(job_id)
        break
    except Exception as e:
        if "Job is in progress" in str(e):
            print(f"Attempt {attempt + 1}/{max_retries}: Still processing...")
            time.sleep(retry_delay)
        else:
            raise

if results is None:
    print("Failed to get results in time.")
    exit(1)

# Print full raw Hume result for inspection (optional)
print(json.dumps(results, indent=4))

from hume_utils import normalize_emotions, combine_surprise_scores

raw_results = []
normalized_results = []

try:
    for entry in results:
        predictions = entry.get("results", {}).get("predictions", [])
        for prediction in predictions:
            prosody_model        = prediction.get("models", {}).get("prosody", {})
            grouped_predictions  = prosody_model.get("grouped_predictions", [])

            for group in grouped_predictions:
                for segment in group.get("predictions", []):
                    emotions   = segment.get("emotions", [])
                    time_frame = segment.get("time", {})
                    midpoint   = (time_frame.get("begin", 0) + time_frame.get("end", 0)) / 2

                    # 1) Build the dict of raw scores (still mixed-case keys)
                    raw_emotions = {
                        emo["name"]: emo["score"]
                        for emo in emotions
                        if emo["name"] in TARGET_EMOTIONS
                    }
                    raw_emotions["time"] = midpoint

                    # 2) Lowercase all keys immediately
                    raw_entry = {k.lower(): v for k, v in raw_emotions.items()}

                    # 3) Combine positive + negative surprise into one 'surprise'
                    if "surprise (positive)" in raw_entry and "surprise (negative)" in raw_entry:
                        raw_entry["surprise"] = combine_surprise_scores(raw_entry)
                        raw_entry.pop("surprise (positive)", None)
                        raw_entry.pop("surprise (negative)", None)
                    else:
                        # if neither channel was present, make sure we still have surprise=0
                        raw_entry.setdefault("surprise", 0.0)

                    # 4) Append to raw_results
                    raw_results.append(raw_entry)

                    # 5) Normalize (this now only ever sees one 'surprise' key)
                    normalized_entry = normalize_emotions(raw_entry)
                    normalized_results.append(normalized_entry)

except Exception as e:
    print("Error while parsing results:", e)
    exit(1)

# … later, after saving per-segment files …

# --- Compute and save averages ---
normalized_avg = compute_average_emotions(normalized_results)
raw_avg        = compute_average_emotions(raw_results)

# --- Combine surprise in the *averages* as well, just in case ---
if "surprise (positive)" in raw_avg and "surprise (negative)" in raw_avg:
    raw_avg["surprise"] = combine_surprise_scores(raw_avg)
    raw_avg.pop("surprise (positive)", None)
    raw_avg.pop("surprise (negative)", None)
else:
    raw_avg.setdefault("surprise", 0.0)

# --- Round and persist ---
normalized_avg_rounded = {k: round(v, 2) for k, v in normalized_avg.items()}
raw_avg_rounded        = {k: round(v, 2) for k, v in raw_avg.items()}

# … (print + json.dump as before) …


# --- Output folders and filenames ---
filtered_results_folder = os.path.join("hume_ai", "filtered_results")
average_results_folder = os.path.join("hume_ai", "filtered_results", "average")
average_raw_folder = os.path.join("hume_ai", "filtered_results", "average_raw")
filtered_folder = os.path.join("hume_ai", "filtered_results", "filtered")
raw_folder = os.path.join("hume_ai", "filtered_results", "raw")

os.makedirs(filtered_results_folder, exist_ok=True)
os.makedirs(average_results_folder, exist_ok=True)
os.makedirs(average_raw_folder, exist_ok=True)
os.makedirs(filtered_folder, exist_ok=True)
os.makedirs(raw_folder, exist_ok=True)

normalized_file = os.path.join(filtered_folder, f"{entry_id}_filtered_emotions.json")
raw_file = os.path.join(raw_folder, f"{entry_id}_raw_emotions.json")
normalized_avg_file = os.path.join(average_results_folder, f"{entry_id}_average_emotions.json")
raw_avg_file = os.path.join(average_raw_folder, f"{entry_id}_average_raw_emotions.json")

# --- Save both per-segment result sets ---
with open(normalized_file, "w") as f:
    json.dump(normalized_results, f, indent=4)
print(f"✅ Normalized emotions saved to '{normalized_file}'")

with open(raw_file, "w") as f:
    json.dump(raw_results, f, indent=4)
print(f"✅ Raw emotions saved to '{raw_file}'")

# --- Compute and save averages ---
normalized_avg = compute_average_emotions(normalized_results)
raw_avg = compute_average_emotions(raw_results)

normalized_avg_rounded = {k: round(v, 2) for k, v in normalized_avg.items()}
raw_avg_rounded = {k: round(v, 2) for k, v in raw_avg.items()}

print("\n Normalized average (proportional, compare with NLP):")
for k, v in normalized_avg_rounded.items():
    print(f"{k}: {v:.3f}")

print("\n Raw average (emotion intensity from voice):")
for k, v in raw_avg_rounded.items():
    print(f"{k}: {v:.3f}")

with open(normalized_avg_file, "w") as f:
    json.dump(normalized_avg_rounded, f, indent=4)
print(f"\n✅ Normalized average saved to '{normalized_avg_file}'")

with open(raw_avg_file, "w") as f:
    json.dump(raw_avg_rounded, f, indent=4)
print(f"✅ Raw average saved to '{raw_avg_file}'")
