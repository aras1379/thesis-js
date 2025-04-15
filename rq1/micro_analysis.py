import json
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import pandas as pd
from scipy.stats import pearsonr
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import active_audio_id, audio_files, emotions_to_analyze 

WINDOW = 5 

def segment_average(time_array, value_array, t_mid, window=WINDOW):
    """
    Computes the average value for a feature (e.g., pitch, intensity) within a window around t_mid.
    """
    mask = (time_array >= (t_mid - window)) & (time_array <= (t_mid + window))
    if np.sum(mask) > 0:
        return np.nanmean(value_array[mask])
    else:
        return np.nan

def get_pitch_data(audio_path):
    """
    Extracts the pitch time series and pitch values from the audio.
    """
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_time = pitch.xs()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan  
    return pitch_time, pitch_values

def get_intensity_data(audio_path):
    """
    Extracts the intensity time series and intensity values from the audio.
    """
    snd = parselmouth.Sound(audio_path)
    intensity = snd.to_intensity()
    intensity_time = intensity.xs()
    intensity_values = np.squeeze(intensity.values.T)
    intensity_values[intensity_values == 0] = np.nan
    return intensity_time, intensity_values

def load_hume_segments(json_path):
    """
    Loads the Hume JSON data that has been pre-processed for segmentation.
    The expected format is a list of dictionaries. Each dictionary has a "time" field
    (the segment midpoint) and emotion scores.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def micro_level_analysis(audio_path, hume_json_path, emotion="Anger"):
    """
    For each segment (as defined in the Hume JSON file), compute the average pitch and intensity
    (using a fixed window around the segment midpoint) and record the Hume emotion score.
    Returns a list of dictionaries with the results.
    """

    pitch_time, pitch_values = get_pitch_data(audio_path)
    intensity_time, intensity_values = get_intensity_data(audio_path)
    
    # Load Hume segment data
    hume_segments = load_hume_segments(hume_json_path)
    
    segments_summary = []
    
    for segment in hume_segments:
        t_mid = segment.get("time", None)
        if t_mid is None:
            continue  
        
        # Compute average pitch and intensity in the time window around t_mid
        avg_pitch = segment_average(pitch_time, pitch_values, t_mid, WINDOW)
        avg_intensity = segment_average(intensity_time, intensity_values, t_mid, WINDOW)
        
        # Get the chosen emotion score from the segment
        emotion_score = segment.get(emotion, np.nan)
        
        # Use fixed keys for consistency in the table
        segment_info = {
            "Segment Mid Time (s)": t_mid,
            "Avg Pitch (Hz)": avg_pitch,
            "Avg Intensity (dB)": avg_intensity,
            f"Hume {emotion} Score": emotion_score
        }
        segments_summary.append(segment_info)
    
    return segments_summary

def plot_combined_scatter(segments_summary, emotion, entry_id):
    """
    Create a single figure with two scatter plots:
    Avg Pitch vs. Emotion and Avg Intensity vs. Emotion.
    """
    emotion_key = f"Hume {emotion} Score"
    
    pitch_vals = np.array([seg["Avg Pitch (Hz)"] for seg in segments_summary])
    intensity_vals = np.array([seg["Avg Intensity (dB)"] for seg in segments_summary])
    emotion_vals = np.array([seg[emotion_key] for seg in segments_summary])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Scatter: Pitch vs Emotion
    axes[0].scatter(pitch_vals, emotion_vals, color='purple')
    axes[0].set_xlabel("Avg Pitch (Hz)")
    axes[0].set_ylabel(f"Hume {emotion} Score")
    axes[0].set_title(f"Avg Pitch vs Hume {emotion} ({entry_id})")
    valid_pitch = ~np.isnan(pitch_vals) & ~np.isnan(emotion_vals)
    if np.sum(valid_pitch) > 1:
        m, b = np.polyfit(pitch_vals[valid_pitch], emotion_vals[valid_pitch], 1)
        axes[0].plot(pitch_vals, m*pitch_vals + b, linestyle='--', color='black')
        r, p = pearsonr(pitch_vals[valid_pitch], emotion_vals[valid_pitch])
        axes[0].legend([f"Pearson r: {r:.2f}, p: {p:.3f}"])

    # Scatter: Intensity vs Emotion
    axes[1].scatter(intensity_vals, emotion_vals, color='darkorange')
    axes[1].set_xlabel("Avg Intensity (dB)")
    axes[1].set_ylabel(f"Hume {emotion} Score")
    axes[1].set_title(f"Avg Intensity vs Hume {emotion} ({entry_id})")
    valid_intensity = ~np.isnan(intensity_vals) & ~np.isnan(emotion_vals)
    if np.sum(valid_intensity) > 1:
        m, b = np.polyfit(intensity_vals[valid_intensity], emotion_vals[valid_intensity], 1)
        axes[1].plot(intensity_vals, m*intensity_vals + b, linestyle='--', color='black')
        r, p = pearsonr(intensity_vals[valid_intensity], emotion_vals[valid_intensity])
        axes[1].legend([f"Pearson r: {r:.2f}, p: {p:.3f}"])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Micro-Level Analysis for Emotion Recognition")
    parser.add_argument("--emotion", type=str, default="all",
                    help="Emotion label(s) to analyze (e.g., Anger, Joy, Sadness, or comma-separated list, or 'all' for all emotions)")

    args = parser.parse_args()

    # Determine which emotions to process.
    if args.emotion.lower() == "all":
        selected_emotions = emotions_to_analyze
    else:
        selected_emotions = [e.strip() for e in args.emotion.split(',')]

    # Set up the file paths.
    entry_id = active_audio_id
    audio_path = audio_files[entry_id]["m4a"]
    hume_json_path = f"hume_ai/filtered_results/filtered/{entry_id}_filtered_emotions.json"
    
    for emotion in selected_emotions:
        segments_summary = micro_level_analysis(audio_path, hume_json_path, emotion=emotion)
        df = pd.DataFrame(segments_summary)
        
        print(f"\nSegment-Level Analysis Table for {emotion}:")
        print(df.to_string(index=False))
        
        # Export to Excel
        excel_filename = f"segment_level_{entry_id}_{emotion.lower()}_analysis.xlsx"

        excel_output_path = os.path.join("exports", excel_filename)
        os.makedirs("exports", exist_ok=True)
        df.to_excel(excel_output_path, index=False)
        print(f"Saved Excel file for {emotion}: {excel_output_path}")
        
        # Plot scatter for Avg Pitch vs. the selected Hume emotion.
        pitch_key = "Avg Pitch (Hz)"
        emotion_key = f"Hume {emotion} Score"
        
        
        # Plot scatter for Avg Intensity vs. the selected Hume emotion.
        intensity_key = "Avg Intensity (dB)"
        plot_combined_scatter(segments_summary, emotion, active_audio_id)



if __name__ == "__main__":
    main()
