# run_compare_vocal_hume.py

import os
import json
from praat_parselmouth.vocal_extract import extract_features
from hume_ai.hume_utils import load_hume_average
from config import active_audio_id, audio_files

entry_id = active_audio_id
audio_path = audio_files[entry_id]["wav"]
file_name = os.path.splitext(os.path.basename(audio_path))[0]
hume_avg_file = f"hume_ai/filtered_results/average_raw/{entry_id}_average_raw_emotions.json"


try:
    # Extract vocal features
    print("Extracting vocal features..")
    vocal_features = extract_features(audio_path)
    print("Vocal features:", vocal_features)

    # Load Hume average emotion scores
    print("loading Hume AI emotion averages...")
    hume_emotions = load_hume_average(hume_avg_file)
    print("Hume Emotions:", hume_emotions)

    # Save both to comparison folder 
    output_data = {
        "entry_id": entry_id,
        "audio_file": audio_path,
        "vocal_features": vocal_features,
        "hume_emotions": hume_emotions
    }

    comparison_folder = "comparisons"
    os.makedirs(comparison_folder, exist_ok=True)
    output_file = os.path.join(comparison_folder, f"{entry_id}_vocal_vs_hume.json")

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n Comparison saved to '{output_file}'")

except Exception as e:
    print("Error during comparison:", e)
