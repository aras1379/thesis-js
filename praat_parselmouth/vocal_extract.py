#vocal_extract.py 

import parselmouth 
from parselmouth.praat import call 
import os 
import numpy as np

def extract_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = float('nan')
    mean_pitch = np.nanmean(pitch_values)
    
    intensity = snd.to_intensity() 
    mean_intensity = np.nanmean(intensity.values)
    
    harmonicity = snd.to_harmonicity()
    hnr_values = harmonicity.values 
    hnr_values = hnr_values[hnr_values != -200]
    mean_hnr = np.nanmean(hnr_values)
    
    formant = snd.to_formant_burg() 
    midpoint = snd.get_total_duration() / 2
    f1 = formant.get_value_at_time(1, midpoint)
    f2 = formant.get_value_at_time(2, midpoint)
    f3 = formant.get_value_at_time(3, midpoint)

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return {
        "duration_seconds": round(snd.get_total_duration(), 2),
        "mean_pitch_hz": round(mean_pitch, 2),
        "mean_intensity_db": round(mean_intensity, 2),
        "mean_hnr_db": round(mean_hnr, 2),
        "formants_hz": {
            "F1": round(f1, 2),
            "F2": round(f2, 2),
            "F3": round(f3, 2),
        },
        "jitter_local": round(jitter, 4),
        "shimmer_local": round(shimmer, 4),
    }
