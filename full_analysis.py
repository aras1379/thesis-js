# full_analysis.py

import os
import json
from nlp_cloud.transcription import transcribe_audio
from nlp_cloud.emotion_analyze import emotion_analyze
from hume_ai.hume_utils import load_hume_average

from save_results import save_combined_result  # 👈 NEW

# Load self-assessed scores (from a central file)
with open("self_assessed/self_scores.json") as f:
    self_scores_all = json.load(f)

# Choose which file/ID to analyze
entry_id = "id_004"
audio_file = "audio_use/negative/4-neg1.m4a"
audio_id = os.path.splitext(os.path.basename(audio_file))[0]
hume_avg_file = f"hume_ai/filtered_results/{audio_id}_average_emotions.json"

try:
    # Step 1: Transcribe
    transcription = transcribe_audio(audio_file)
    print("📝 Transcription:", transcription)

    # Step 2: NLP Cloud Emotion Analysis
    nlp_emotions = emotion_analyze(transcription)
    print("📊 NLP Emotions:", nlp_emotions)

    # Step 3: Load Hume AI results
    hume_emotions = load_hume_average(hume_avg_file)
    print("🎧 Hume Emotions:", hume_emotions)

    # Step 4: Get self-assessed scores
    self_assessed = self_scores_all.get(entry_id, {})
    if not self_assessed:
        print(f"⚠️ No self-assessed scores found for {entry_id}")

    # Step 5: Combine and save using helper function
    result_data = {
        "audio_file": audio_file,
        "transcription": transcription,
        "nlp_emotions": nlp_emotions,
        "hume_emotions": hume_emotions,
        "self_assessed": self_assessed
    }

    save_combined_result(entry_id, result_data)

except Exception as e:
    print("❌ Error during full analysis:", e)