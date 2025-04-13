# run_analysis.py 

from transcription import transcribe_audio 
from emotion_analyze import emotion_analyze
from config import audio_files

audio_path = audio_files["id_004_neg"]["wav"]

try:
    transcription = transcribe_audio(audio_path)
    print("Transcription: ", transcription)
    
    emotion = emotion_analyze(transcription)
    print("Emotion analysis: ", emotion)

except Exception as e: 
    print("Error: ", e)