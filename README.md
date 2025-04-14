# thesis-js

## To do: 

Automatisera konvertering från mp4 till wav 

Skriva om i rapport: 
Hur har normaliserat hume values 

Pearson och p values ? 

Surprise neg + pos = surprise 

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
pip3 install -r requirements.txt 

Change which clip to analyze in "config.py" 
-------------------- NEW DATA ------------------------
Change clip in config 
1. run python3 run_all.py
2. run python3 run_compare_vocal_hume.py 

-------------------- FOR RQ1 ------------------------

Overlay_visualization.py: 
- Analyzes 1 file and gives diagram for one emotion (or more, opens after each other) 
- uses praat + hume values over time 

run: 
python3 rq1/overlay_visualization.py --emotion "anger" 
or: --emotion all 

scatter_plot.py 
- load aggregated clip data from "comparisons" folder 
- uses functions from data_utils.py 
- creates scatter plots that show relationships between acoustic features and emotion acores 

run: 
python3 rq1/scatter_plot.py --emotion "Joy,Sadness" 
or: --emotion all

correlation_table.py
- loads aggregated data for multiple clips from "comparisons" folder 
- computes Pearson correlation and p-values for a series of acoustic features and emotion pairs 
- stored in a table that is printed in terminal 

run:
python3 rq1/correlation_table.py --emotion all

micro_analysis.py
- perform micro-level analysis on single clip 
- extracts time-series acoustic features (pitch and intensity) from audio file 
- aligns these time series with time-stamped emotion scores from hume json file that INCLUDE TIME 
- shows result in window 
- prints summary table with Pandas in terminal 

run:
python3 rq1/micro_analysis.py --emotion "Anger"
or: --emotion all



-------------------- FOR RQ2 ------------------------
emotion_comparison_bar.py 
- visualize side by side comparison on emotions for a single clip 
- loads results from the file: results_combined.json (hume + nlp)

run:
python3 rq2/emotion_comparison_bar.py --clip id_004_pos --emotion all


emotion_comparison_correlation.py 
- compare performance of hume and nlp across all clips 
- loads from results_combined.json 
- pearson correlation (r) and p.value 
- presented in table in terminal using Pandas 

run: 
python3 rq2/emotion_comparison_correlation.py --emotion all
or: --emotion "Joy,Sadness" 



----- FOLDER STRUCTURE -------
THESIS_JS/
├── README.md                
├── __pycache__/             
├── audio/                   
├── audio_use/               # Processed audio files split by type (e.g., negative and positive).
│   ├── negative/            # Audio files from negative interviews (e.g., 1-neg.m4a, 1-neg.wav).
│   └── positive/            # Audio files from positive interviews (e.g., 1-pos.m4a, 1-pos.wav).
├── comparisons/             # Aggregated/combined JSON files from your analysis (e.g., vocal_vs_hume results).
├── config.py                # Configuration file containing settings such as active_audio_id, file paths, and emotion labels.
├── data_utils.py            # Utility functions for loading, processing, and aggregating JSON data.
├── full_analysis.py         # A script (or possibly a wrapper) that runs the entire analysis pipeline.
├── hume_ai/                 # Hume AI-related code.
│   ├── __pycache__/         # Compiled cache for Hume modules.
│   ├── average_functions.py  # Functions for computing average emotion scores from Hume outputs.
│   ├── emotion_recognition.py  # Code for performing emotion recognition using Hume.
│   ├── fetch_results.py     # Code for fetching Hume API results.
│   ├── filtered_results/    # Folder where filtered/normalized Hume output JSON files are stored.
│   ├── hume_utils.py        # Hume utility functions (e.g., normalization, combining surprise scores).
│   └── run_emotion_analysis.py  # Script to run emotion analysis using Hume AI.
├── nlp_cloud/               # NLP Cloud (text-based emotion recognition) related code.
│   ├── __pycache__/         
│   ├── emotion_analyze.py   # Script to analyze emotions from text using NLP Cloud.
│   ├── run_analysis.py      # Script to run the full analysis with NLP Cloud.
│   └── transcription.py     # Script for generating transcripts if applicable.
├── praat_parselmouth/       # Code related to extracting vocal features using Praat/Parselmouth.
│   ├── __pycache__/         
│   ├── run_vocal_extract.py # Script to extract vocal features from audio files.
│   ├── visualize_features.py  # Contains plotting functions for vocal features (e.g., pitch, intensity, formants).
│   └── vocal_extract.py     # Functions to extract acoustic features from audio using Parselmouth.
├── requirements.txt         # List of all required Python packages to run this project.
├── results_combined.json    # Combined JSON file with analysis results from multiple clips.
├── rq1/                     # Scripts and analysis for Research Question 1 (Speech-Based emotion recognition and vocal markers).
│   ├── correlation_table.py    # Creates a correlation table between acoustic features and Hume emotion scores.
│   ├── micro_analysis.py       # Performs micro-level (segment-by-segment) analysis of audio and Hume data.
│   ├── overlay_visualization.py# Overlays vocal feature time-series with Hume emotion scores.
│   └── scatter_plot.py         # Generates scatter plots for aggregated acoustic features vs. Hume emotion scores.
├── rq2/                     # Scripts and analysis for Research Question 2 (Text vs. Speech emotion recognition).
│   ├── emotion_comparison_bar.py       # Creates grouped bar charts comparing NLP Cloud (text) vs. Hume (speech) emotion scores.
│   └── emotion_comparison_correlation.py # Performs correlation analysis between text-based and speech-based emotion scores.
├── run_all.py               # Script to run all components of the analysis pipeline.
├── run_compare_vocal_hume.py# Script to compare extracted vocal features with Hume outputs.
├── save_results.py          # Utility script to save analysis results.
└── self_assessed/           # Contains self-reported emotion assessments.
    └── self_scores.json     # JSON file with self-assessed emotion ratings.
