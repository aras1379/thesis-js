# full_analysis.py

import os
import json
from nlp_cloud.transcription import transcribe_audio
from nlp_cloud.emotion_analyze import emotion_analyze
from hume_ai.hume_utils import load_hume_average, normalize_emotions

from save_results import save_combined_result 
from config import active_audio_id, audio_files

# Load self-assessed scores 
with open("self_assessed/self_scores.json") as f:
    self_scores_all = json.load(f)

entry_id = active_audio_id
audio_path = audio_files[entry_id]["m4a"]
file_name = os.path.splitext(os.path.basename(audio_path))[0]

hume_avg_file = f"hume_ai/filtered_results/average/{entry_id}_average_emotions.json"

try:
    #Transcribe
    transcription = transcribe_audio(audio_path)
    print("Transcription:", transcription)

    #NLP Cloud 
    raw_nlp = emotion_analyze(transcription)
    nlp_norm = normalize_emotions(raw_nlp)
    nlp_rounded = {k: round(v, 2) for k, v in nlp_norm.items()}

    # Load Hume results
    hume_emotions = load_hume_average(hume_avg_file)
    print("Hume Emotions:", hume_emotions)

    # Get self-assessed labels 
    entry_self = self_scores_all.get(entry_id, {})
    raw_self  = entry_self.get("self_assessed", {})
    if not raw_self:
        print(f"No self-assessed scores found for {entry_id}")
    # Now normalize that flat map of 5 numbers
    self_emotions = normalize_emotions(raw_self)
    self_emotions = {k: round(v, 2) for k, v in self_emotions.items()}


    # save data
    result_data = {
        "audio_file": audio_path,
        "transcription": transcription,
        "nlp_emotions": nlp_rounded,
        "hume_emotions": hume_emotions,
        "self_assessed": self_emotions
    }
    save_combined_result(entry_id, result_data)

except Exception as e:
    print("Error during full analysis:", e)