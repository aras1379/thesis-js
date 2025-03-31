# run_analysis.py 

from transcription import transcribe_audio 
from emotion_analyze import emotion_analyze

audio_path = "audio/s+j-clip.m4a"

try:
    transcription = transcribe_audio(audio_path)
    print("Transcription: ", transcription)
    
    emotion = emotion_analyze(transcription)
    print("Emotion analysis: ", emotion)

except Exception as e: 
    print("Error: ", e)