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
    
    print(f"\n File: {os.path.basename(audio_path)}")
    print(f"Duration: {snd.get_total_duration():.2f} seconds")
    print(f"Mean Pitch: {mean_pitch:.2f} Hz")
    print(f"Mean Intensity: {mean_intensity:.2f} dB")
    print(f"Harmonics-to-Noise Ratio (Mean HNR): {mean_hnr:.2f} dB")
    
    midpoint = snd.get_total_duration() / 2
    f1 = formant.get_value_at_time(1, midpoint)
    f2 = formant.get_value_at_time(2, midpoint)
    f3 = formant.get_value_at_time(3, midpoint)
    print(f"Formants at midpoint: ({midpoint:.2f}s): F1 = {f1:.2f} Hz, F2 = {f2:.2f} Hz, F3 = {f3:.2f} Hz")
    
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    print(f"Jitter (local): {jitter:.4f}")
    print(f"Shimmer (local): {shimmer:.4f}")

if __name__ == "__main__":
    extract_features("audio/s+j-clipwav.wav")