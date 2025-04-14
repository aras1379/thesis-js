# run_analysis.py 

from transcription import transcribe_audio 
from emotion_analyze import emotion_analyze
from config import audio_files, active_audio_id

entry_id = active_audio_id
audio_path = audio_files[entry_id]["wav"]

try:
    transcription = transcribe_audio(audio_path)
    print("Transcription: ", transcription)
    
    emotion = emotion_analyze(transcription)
    print("Emotion analysis: ", emotion)

except Exception as e: 
    print("Error: ", e)