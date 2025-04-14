# thesis-js

## To do: 

Automatisera konvertering från mp4 till wav 

Skriva om i rapport: 
Metod:
- Hur har normaliserat hume values 

- Pearson och p values ? 

- Surprise neg + pos = surprise 

- Hur vi ska jämföra svaren 

Theoretical framework: 
- NLP Cloud 

## Contents 
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [How to Run the Code](#how-to-run-the-code)
  - [General Pipeline](#general-pipeline)
  - [RQ1: Speech-Based Emotion Recognition & Vocal Markers](#rq1)
  - [RQ2: Text-Based vs. Speech-Based Emotion Recognition](#rq2)
- [Folder Structure](#folder-structure)
- [File Descriptions](#file-descriptions)
- [Additional Notes](#additional-notes)


## Dependencies 
```bash
pip3 install -r requirements.txt 
```

## Configuration 
Change which clip to analyze in "config.py" 
**------ NEW DATA ------**
Change clip in config: active_audio_id 
1. run python3 run_all.py
2. run python3 run_compare_vocal_hume.py 

## How to run code and files purpose 
## RQ1 
Use filtered_results/ xxx_raw_emotions.json, save in comparisons/
### - Overlay_visualization.py: 
- Analyzes 1 file and gives diagram for one emotion (or more, opens after each other) 
- uses praat + hume values over time 

```bash 
python3 rq1/overlay_visualization.py --emotion "anger"
#or: --emotion all
```

### - scatter_plot.py 
- load aggregated clip data from "comparisons" folder 
- uses functions from data_utils.py 
- creates scatter plots that show relationships between acoustic features and emotion acores 

```bash 
python3 rq1/scatter_plot.py --emotion "Joy,Sadness" 
#or: --emotion all
```

### - correlation_table.py
- loads aggregated data for multiple clips from "comparisons" folder 
- computes Pearson correlation and p-values for a series of acoustic features and emotion pairs 
- stored in a table that is printed in terminal 

```bash 
python3 rq1/correlation_table.py --emotion all
```

### - micro_analysis.py
- perform micro-level analysis on single clip 
- extracts time-series acoustic features (pitch and intensity) from audio file 
- aligns these time series with time-stamped emotion scores from hume json file that 
- shows result in window 
- prints summary table with Pandas in terminal 

```bash 
python3 rq1/micro_analysis.py --emotion "Anger"
#or: --emotion all
```

## RQ2
Use filtered_results/ xxx_average_emotions, saved in results_combined.json  
### - emotion_comparison_bar.py 
- visualize side by side comparison on emotions for a single clip 
- loads results from the file: results_combined.json (hume + nlp)

```bash
python3 rq2/emotion_comparison_bar.py --clip id_004_pos --emotion all`
#change clip id
```
### - emotion_comparison_correlation.py 
- compare performance of hume and nlp across all clips 
- loads from results_combined.json 
- pearson correlation (r) and p.value 
- presented in table in terminal using Pandas 

```bash
python3 rq2/emotion_comparison_correlation.py --emotion all
# or: --emotion "Joy,Sadness" 
```

## File Description 
```config.py```: Contains configuration settings (clip IDs, paths, emotion labels).

data_utils.py: Utilities for loading and processing JSON data.

full_analysis.py: Script to run the complete analysis pipeline.

Hume AI Folder (hume_ai/):

Scripts for emotion recognition using Hume, handling API results, and processing Hume output.

NLP Cloud Folder (nlp_cloud/):

Contains code for text-based emotion analysis and optional transcript generation.

Praat/Parselmouth Folder (praat_parselmouth/):

Extracts and visualizes vocal features like pitch and intensity.

RQ1 Folder (rq1/):

Contains scripts for visualizing, analyzing, and correlating speech-based emotion scores.

RQ2 Folder (rq2/):

Contains scripts comparing text-based emotion recognition with Hume (speech-based) results.

Utility Scripts:

run_all.py: Executes the entire pipeline.

run_compare_vocal_hume.py: Compares vocal features with Hume outputs.

save_results.py: Saves analysis outcomes.