#data_utils.py 
import json
import glob
import numpy as np
from config import emotions_to_analyze  

import json
import os

ACOUSTIC_STATS_PATH = os.path.join(os.path.dirname(__file__), "acoustic_stats.json")

with open(ACOUSTIC_STATS_PATH, "r") as f:
    ACOUSTIC_STATS = json.load(f)

def z_score(value, mean, std):
    try:
        return (value - mean) / std if std > 0 else 0
    except:
        return 0

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_clip_data(file_path):
    """
    Loads the aggregated data from a clip file.
    Expected JSON format:
    {
      "entry_id": ...,
      "audio_file": ...,
      "vocal_features": {
         "mean_pitch_hz": ...,
         "mean_intensity_db": ...,
         "formants_hz": {"F1": ..., "F2": ..., "F3": ...},
         ... (other features)
      },
      "hume_emotions": {
         "Anger": ...,
         "Disgust": ...,
         "Fear": ...,
         "Joy": ...,
         "Sadness": ...,
         "Surprise (negative)": ...,
         "Surprise (positive)": ...
      }
    }
    """
    data = load_json(file_path)
    vocal = data.get("vocal_features", {})
    hume = data.get("hume_emotions", {})
    return vocal, hume

def load_all_clip_data(directory):
    """
    Loads all clip data from a specified folder (directory).
    Returns a list of tuples: (vocal_data, hume_data) for each file.
    """
    files = sorted(glob.glob(f"{directory}/*.json"))
    data_list = []
    for file in files:
        vocal, hume = load_clip_data(file)
        data_list.append((vocal, hume))
    return data_list

def compute_feature_arrays(data_list):
    """
    Given a list of (vocal, hume) tuples, extracts arrays of aggregated features,
    including all emotion scores specified in the config.
    """
    mean_pitches = []
    mean_intensities = []
    f1_values, f2_values, f3_values = [], [], []
    
    emotion_values = {emotion.lower(): [] for emotion in emotions_to_analyze}
    
    for vocal, hume in data_list:
        mean_pitches.append(vocal.get("mean_pitch_hz", np.nan))
        mean_intensities.append(vocal.get("mean_intensity_db", np.nan))
        
        formants = vocal.get("formants_hz", {})
        f1_values.append(formants.get("F1", np.nan))
        f2_values.append(formants.get("F2", np.nan))
        f3_values.append(formants.get("F3", np.nan))
        
 
        for emotion in emotions_to_analyze:
            value = hume.get(emotion, np.nan)
            emotion_values[emotion.lower()].append(value)
    
    # Convert all lists to numpy arrays.
    return {
        "mean_pitch": np.array(mean_pitches),
        "mean_intensity": np.array(mean_intensities),
        "F1": np.array(f1_values),
        "F2": np.array(f2_values),
        "F3": np.array(f3_values),
        **{emotion: np.array(values) for emotion, values in emotion_values.items()}
    }
