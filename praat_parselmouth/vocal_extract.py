## Extract vocal features to tables 
## This file is used for RQ1 folder 

import numpy as np
import parselmouth
from parselmouth.praat import call

def estimate_voiced_unvoiced(snd, threshold_db=50.0):
    intensity = snd.to_intensity()
    dt = intensity.dx
    voiced = (intensity.values[0] > threshold_db).sum() * dt
    return voiced, snd.get_total_duration() - voiced

def extract_features(audio_path):
    if isinstance(audio_path, str):
        snd = parselmouth.Sound(audio_path)
    else:
        snd = audio_path
   

    # 1) pitch → semitones above 150 Hz
    pitch = snd.to_pitch()
    freqs = pitch.selected_array['frequency']
    freqs[freqs == 0] = np.nan
    mean_hz = float(np.nanmean(freqs))
    mean_pitch_st = 12 * np.log2(mean_hz / 150.0)
    mean_pitch_hz = round(mean_hz, 2)

    # 2) intensity (dB)
    intensity = snd.to_intensity()
    mean_intensity_db = float(np.nanmean(intensity.values))

    # 3) HNR (dB)
    hnr = snd.to_harmonicity_ac(0.01, 75, 0.1, 1.0)
    hnr_v = hnr.values.flatten()
    hnr_v = hnr_v[hnr_v != -200]
    mean_hnr_db = float(np.nanmean(hnr_v))

    # 4) jitter & shimmer
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local  = float(call(pp,       "Get jitter (local)",        0,0,0.0001,0.02,1.3))
    shimmer_local = float(call([snd, pp], "Get shimmer (local)", 0,0,0.0001,0.02,1.3,1.6))

    # 5) formants & bandwidths at midpoint
    mid  = snd.get_total_duration() / 2
    form = snd.to_formant_burg()
    f1 = form.get_value_at_time(1, mid);   b1 = form.get_bandwidth_at_time(1, mid)
    f2 = form.get_value_at_time(2, mid);   b2 = form.get_bandwidth_at_time(2, mid)
    f3 = form.get_value_at_time(3, mid);   b3 = form.get_bandwidth_at_time(3, mid)

    # 6) voiced/unvoiced durations
    voiced_length, unvoiced_length = estimate_voiced_unvoiced(snd)

    return {
        "mean_pitch_st":     round(mean_pitch_st,    2),
        "mean_pitch_hz":     mean_pitch_hz,
        "mean_intensity_db": round(mean_intensity_db,2),
        "mean_hnr_db":       round(mean_hnr_db,      2),
        "jitter_local":      round(jitter_local,     4),
        "shimmer_local":     round(shimmer_local,    4),
        # group formants into a sub‐dict:
        "formants_hz": {
            "F1": round(f1, 2),
            "F2": round(f2, 2),
            "F3": round(f3, 2),
            "B1": round(b1, 2),
            "B2": round(b2, 2),
            "B3": round(b3, 2),
        },
        "voiced_length":     round(voiced_length,    2),
        "unvoiced_length":   round(unvoiced_length,  2),
    }
