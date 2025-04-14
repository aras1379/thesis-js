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

