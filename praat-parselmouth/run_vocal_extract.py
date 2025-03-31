import parselmouth 
from parselmouth.praat import call 
from visualize_features import visualize_all_features
from visualize_features import visualize_pitch
from visualize_features import visualize_intensity 
from visualize_features import visualize_formant 
from vocal_extract import extract_features 

audio_path = "audio/s+j-clipwav.wav"
snd = parselmouth.Sound(audio_path)
try:
    vocal_extraction = extract_features(audio_path)
    
    pitch_visual = visualize_pitch(snd)
    intensity_visual = visualize_intensity(snd)
    formant_visual = visualize_formant(snd)
    
    full_visualize = visualize_all_features(snd)
    
except Exception as e: 
    print("Error: ", e)