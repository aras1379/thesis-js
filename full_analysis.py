# full_analysis.py

import os
import json
from nlp_cloud.transcription import transcribe_audio
from nlp_cloud.emotion_analyze import emotion_analyze
from hume_ai.hume_utils import load_hume_average

from save_results import save_combined_result  # 👈 NEW
from config import active_audio_id, audio_files

# Load self-assessed scores 
with open("self_assessed/self_scores.json") as f:
    self_scores_all = json.load(f)

entry_id = active_audio_id
audio_path = audio_files[entry_id]["m4a"]
file_name = os.path.splitext(os.path.basename(audio_path))[0]

hume_avg_file = f"hume_ai/filtered_results/{entry_id}_average_emotions.json"

try:
    #Transcribe
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)

    #NLP Cloud 
    nlp_emotions = emotion_analyze(transcription)
    print("NLP Emotions:", nlp_emotions)

    # Load Hume results
    hume_emotions = load_hume_average(hume_avg_file)
    print("Hume Emotions:", hume_emotions)

    # Get self-assessed labels 
    self_assessed = self_scores_all.get(entry_id, {})
    if not self_assessed:
        print(f"No self-assessed scores found for {entry_id}")

    # save data
    result_data = {
        "audio_file": audio_path,
        "transcription": transcription,
        "nlp_emotions": nlp_emotions,
        "hume_emotions": hume_emotions,
        "self_assessed": self_assessed
    }
    save_combined_result(entry_id, result_data)

except Exception as e:
    print("Error during full analysis:", e)