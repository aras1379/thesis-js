#run_vocal_extract.py 

import parselmouth 
from parselmouth.praat import call 
from visualize_features import visualize_all_features
from visualize_features import visualize_pitch
from visualize_features import visualize_intensity 
from visualize_features import visualize_formant 
from vocal_extract import extract_features 
from config import audio_files, active_audio_id

entry_id = active_audio_id
audio_path = audio_files[entry_id]["wav"]

snd = parselmouth.Sound(audio_path)
try:
    vocal_extraction = extract_features(audio_path)
    
    pitch_visual = visualize_pitch(snd)
    intensity_visual = visualize_intensity(snd)
    formant_visual = visualize_formant(snd)
    
    full_visualize = visualize_all_features(snd)
    
except Exception as e: 
    print("Error: ", e)