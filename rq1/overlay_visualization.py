import json
import numpy as np 
import matplotlib.pyplot as plt 
import parselmouth 
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze

def get_pitch_data(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_time = pitch.xs()  
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    return pitch_time, pitch_values

def get_intensity_data(audio_path):
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    intensity_time = intensity.xs()
    intensity_values = np.squeeze(intensity.values.T)
    intensity_values[intensity_values == 0] = np.nan
    return intensity_time, intensity_values

def load_hume_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def overlay_plot(audio_path, hume_json_path, emotion="anger"):
    # Extract Praat feature data
    pitch_time, pitch_values = get_pitch_data(audio_path)
    intensity_time, intensity_values = get_intensity_data(audio_path)
    
    # Load Hume emotion data 
    hume_data = load_hume_data(hume_json_path)
    hume_time = [entry["time"] for entry in hume_data]
    hume_emotion = [entry.get(emotion, np.nan) for entry in hume_data]
    
    # Create a figure with two subplots (one for pitch and one for intensity)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
    
    ### Top subplot: Pitch overlay with Hume emotion
    entry_id = active_audio_id
    audio_path = audio_files[entry_id]["m4a"]
    ax1 = axes[0]
    ax1.plot(pitch_time, pitch_values, label='Pitch (Hz)', color='blue')
    ax1.set_ylabel('Pitch (Hz)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title(f"Pitch and Hume {emotion} Over Time {active_audio_id}")
    
    # Secondary y-axis for Hume emotion
    ax1b = ax1.twinx()
    ax1b.plot(hume_time, hume_emotion, marker='o', linestyle='--', label=f'Hume {emotion}', color='red')
    ax1b.set_ylabel(f'Hume {emotion} Score', color='red')
    ax1b.tick_params(axis='y', labelcolor='red')
    
    ### Bottom subplot: Intensity overlay with Hume emotion
    ax2 = axes[1]
    ax2.plot(intensity_time, intensity_values, label='Intensity (dB)', color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Intensity (dB)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_title(f"Intensity and Hume {emotion} Over Time")
    
    # Secondary y-axis for Hume emotion
    ax2b = ax2.twinx()
    ax2b.plot(hume_time, hume_emotion, marker='o', linestyle='--', label=f'Hume {emotion}', color='red')
    ax2b.set_ylabel(f'Hume {emotion} Score', color='red')
    ax2b.tick_params(axis='y', labelcolor='red')
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay Visualization for Emotion Recognition")
    parser.add_argument("--emotion", type=str, default="all",
                        help="Emotion label(s) to analyze (e.g., Anger, Joy, Sadness, or comma-separated list, or 'all' for all emotions)")
    args = parser.parse_args()
    
    # Determine the selected emotion(s)
    if args.emotion.lower() == "all":
        selected_emotions = emotions_to_analyze
    else:
        selected_emotions = [e.strip() for e in args.emotion.split(',')]
    
    # Set up file paths.
    entry_id = active_audio_id
    audio_path = audio_files[entry_id]["m4a"]
    hume_json_path = f"hume_ai/filtered_results/filtered/{entry_id}_filtered_emotions.json"
    
    for emotion in selected_emotions:
        overlay_plot(audio_path, hume_json_path, emotion=emotion)
