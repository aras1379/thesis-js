## Compare hume value scores time to time vs pitch and intensity
## can add more vocal features 
## ANVÄND 
import os
import json
import sys
import pandas as pd
import parselmouth
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# make sure your project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import plot_and_save
from config import active_audio_id, audio_files
from rq1.micro_analysis import micro_level_analysis_all  # adjust import if needed
TARGET_EMO = "joy"
WINDOW = 3  # seconds around each Hume timestamp

def segment_average(time_array, value_array, t_mid, window=WINDOW):
    mask = (time_array >= (t_mid - window)) & (time_array <= (t_mid + window))
    return np.nanmean(value_array[mask]) if mask.any() else np.nan

def get_pitch_intensity_series(wav_path):
    snd = parselmouth.Sound(wav_path)
    # pitch
    p = snd.to_pitch()
    t_pitch = p.xs()
    v_pitch = p.selected_array["frequency"]
    v_pitch[v_pitch == 0] = np.nan
    # intensity
    I = snd.to_intensity()
    t_int = I.xs()
    v_int = I.values[0]
    v_int[v_int == 0] = np.nan
    return (t_pitch, v_pitch), (t_int, v_int)

def acoustic_vs_hume_time_series(wav_path, hume_json_path, target_emo):
    """
    For each Hume timestamp, compute average pitch & intensity in ±WINDOW,
    pull that emotion's probability, and plot both over time plus scatter.
    """
    # load Hume time‐series
    with open(hume_json_path) as f:
        hume_segments = json.load(f)

    # get full pitch/intensity series
    (t_pitch, v_pitch), (t_int, v_int) = get_pitch_intensity_series(wav_path)

    rows = []
    for seg in hume_segments:
        t_mid = seg.get("time")
        if t_mid is None:
            continue
        rows.append({
            "SegmentTime":     t_mid,
            "AvgPitch_Hz":     segment_average(t_pitch, v_pitch, t_mid),
            "AvgIntensity_dB": segment_average(t_int,  v_int,  t_mid),
            f"Hume_{target_emo}": seg.get(target_emo, np.nan)
        })

    df = pd.DataFrame(rows).sort_values("SegmentTime")

    # --- make plots ---
    emo_col = f"Hume_{target_emo}"
    entry_id = os.path.splitext(os.path.basename(wav_path))[0]
    
    save_dir = "plots"

    # 1) time series: pitch + emotion
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["SegmentTime"], df["AvgPitch_Hz"], label="Pitch (Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pitch (Hz)")
    ax2 = ax1.twinx()
    ax2.plot(df["SegmentTime"], df[emo_col], color="C1", label=target_emo.title())
    ax2.set_ylabel(f"Hume {target_emo.title()} prob")
    fig.suptitle(f"Pitch & {target_emo.title()} over time: {entry_id}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    
    plot_and_save(fig, f"{save_dir}/pitch_{target_emo}_{entry_id}")

    # 2) time series: intensity + emotion
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df["SegmentTime"], df["AvgIntensity_dB"], label="Intensity (dB)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Intensity (dB)")
    ax2 = ax1.twinx()
    ax2.plot(df["SegmentTime"], df[emo_col], color="C1", label=target_emo.title())
    ax2.set_ylabel(f"Hume {target_emo.title()} prob")
    fig.suptitle(f"Intensity & {target_emo.title()} over time: {entry_id}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plot_and_save(fig, f"{save_dir}/intensity_{target_emo}_{entry_id}")

    # 3) scatter: AvgPitch vs emotion, AvgIntensity vs emotion
    fig, (axp, axi) = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(x="AvgPitch_Hz",     y=emo_col, data=df, ax=axp)
    axp.set_title("Pitch vs emotion")
    sns.scatterplot(x="AvgIntensity_dB", y=emo_col, data=df, ax=axi)
    axi.set_title("Intensity vs emotion")
    plt.suptitle(f"{target_emo.title()} prob vs acoustic features")
    plt.tight_layout()
    plt.show()
    

    
def main():
    entry_id      = active_audio_id
    wav_path      = audio_files[entry_id]["m4a"]
    hume_json     = f"hume_ai/filtered_results/filtered/{entry_id}_filtered_emotions.json"
    target_emotion = "joy"

    acoustic_vs_hume_time_series(wav_path, hume_json, target_emotion)
    
if __name__ == '__main__':
    main()
