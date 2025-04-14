#visualize_features.py

import parselmouth 
from parselmouth.praat import call 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set_theme() 
plt.rcParams['figure.dpi'] = 120 

def draw_pitch(pitch):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan 
    plt.plot(pitch.xs(), pitch_values, label = "Pitch (Hz)", color="blue")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, label="Intensity (dB)", color="orange")

def draw_formants(formant, snd):
    t = formant.xs() 
    for i in range(1, 4): 
        formant_values = [formant.get_value_at_time(i, time) for time in t]
        plt.plot(t, formant_values, label=f"F{i}", linestyle="--")

def visualize_single_feature(snd, extractor_fn, drawer_fn, title, ylabel):
    data = extractor_fn()
    plt.figure(figsize=(10,6))
    drawer_fn(data)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.legend() 
    plt.title(title)
    plt.tight_layout()
    plt.show() 
    
def visualize_pitch(snd): 
    visualize_single_feature(snd, lambda: snd.to_pitch(), draw_pitch, "Pitch over time", "Frequency(Hz)")
    
    
def visualize_intensity(snd): 
    visualize_single_feature(snd, lambda: snd.to_intensity(), draw_intensity, "Intensity over time", "dB")

def visualize_formant(snd): 
    visualize_single_feature(snd, lambda: snd.to_formant_burg(), lambda f: draw_formants(f, snd), "Formants over time", "Frequency(Hz)")

#Visualize all features        
def visualize_all_features(snd):
    pitch = snd.to_pitch() 
    intensity = snd.to_intensity() 
    formant = snd.to_formant_burg() 
    
    plt.figure(figsize=(10,6))
    draw_pitch(pitch)
    draw_intensity(intensity)
    draw_formants(formant, snd)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency / Intensity")
    plt.legend()
    plt.title("Vocal features over time")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    snd = parselmouth.Sound("audio/s+j-clipwav.wav")
    visualize_pitch(snd)
    visualize_intensity(snd)
    visualize_formant(snd)
    visualize_all_features(snd)
    
