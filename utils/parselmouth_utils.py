import os
import parselmouth
import pandas as pd
from praat_parselmouth.vocal_extract import extract_features


def segment_and_extract_features(audio_input, segment_length: float) -> pd.DataFrame:

    
    # Load Sound if necessary
    if isinstance(audio_input, str):
        snd = parselmouth.Sound(audio_input)
    else:
        snd = audio_input

    duration = snd.get_total_duration()
    segments = []
    t = 0.0
    while t < duration:
        seg_dur = min(segment_length, duration - t)
        if seg_dur < 0.07:  # minimal duration threshold
            break

        snippet = snd.extract_part(t, t + seg_dur, preserve_times=False)
        feats = extract_features(snippet)

        # Flatten formants dict into top-level columns
        formants = feats.pop("formants_hz", {})
        for form_label, value in formants.items():
            feats[f"Formant_{form_label}_Hz"] = value

        feats["Segment_Start"] = round(t, 4)
        segments.append(feats)
        t += segment_length

    return pd.DataFrame(segments)
