#config.py 
INPUT_RQ2_RQ3 = "results_combined_rq2_rq3.json"
INPUT_DIR_OLD = "comparisons_rq1_2"
INPUT_DIR_V3 = "new_rq_run_V4"
INPUT_DIR_V0 = "new_rq_run_V0"
EXPORT_DIR = "THESIS_NEW"
PLOT_DIR = "THESIS_PLOTS"

#Change this to the current audio for analyzation 
active_audio_id = "id_006_neg"

emotions_to_analyze = [
    "anger", "joy", "sadness", "surprise (positive)", 
    "surprise (negative)", "fear"
]

audio_files = {
    "id_001_neg": {
        "m4a": "audio_use/negative/1-neg.m4a",
        "wav": "audio_use/negative/1-neg.wav"
    },
    "id_002_neg":{
        "m4a": "audio_use/negative/2-neg.m4a",
        "wav": "audio_use/negative/2-neg.wav"
    },
  
    "id_004_neg": {
        "m4a": "audio_use/negative/4-neg1.m4a",
        "wav": "audio_use/negative/4-neg1.wav"
    },

    "id_005_neg": {
        "m4a": "audio_use/negative/5-neg.m4a",
        "wav": "audio_use/negative/5-neg.wav"
    },

    "id_006_neg": {
        "m4a": "audio_use/negative/6-neg.m4a",
        "wav": "audio_use/negative/6-neg.wav"
    },
    "id_007_neg": {
        "m4a": "audio_use/negative/7-neg.m4a",
        "wav": "audio_use/negative/7-neg.wav"
    },
    "id_008_neg": {
        "m4a": "audio_use/negative/8-neg.m4a",
        "wav": "audio_use/negative/8-neg.wav"
    },
    "id_009_neg": {
        "m4a": "audio_use/negative/9-neg.m4a",
        "wav": "audio_use/negative/9-neg.wav"
    },
    "id_010_neg":{
        "m4a": "audio_use/negative/10-neg.m4a",
        "wav": "audio_use/negative/10-neg.wav"
    },
    "id_011_neg":{
        "m4a": "audio_use/negative/11-neg.m4a",
        "wav": "audio_use/negative/11-neg.wav"
    },
    "id_012_neg":{
        "m4a": "audio_use/negative/12-neg.m4a",
        "wav": "audio_use/negative/12-neg.wav"
    },
    "id_013_neg":{
        "m4a": "audio_use/negative/13-neg.m4a",
        "wav": "audio_use/negative/13-neg.wav"
    },
    "id_014_neg":{
        "m4a": "audio_use/negative/14-neg.m4a",
        "wav": "audio_use/negative/14-neg.wav"
    },
    "id_015_neg":{
        "m4a": "audio_use/negative/15-neg.m4a",
        "wav": "audio_use/negative/15-neg.wav"
    },
    "id_016_neg":{
        "m4a": "audio_use/negative/16-neg.m4a",
        "wav": "audio_use/negative/16-neg.wav"
    },
    
    #POSITIVE 
    "id_001_pos": {
        "m4a": "audio_use/positive/1-pos.m4a",
        "wav": "audio_use/positive/1-pos.wav"
    },
    "id_002_pos": {
        "m4a": "audio_use/positive/2-pos.m4a",
        "wav": "audio_use/positive/2-pos.wav"
    },
 
    "id_004_pos": {
        "m4a": "audio_use/positive/4-pos.m4a",
        "wav": "audio_use/positive/4-pos.wav"
    },
    "id_005_pos": {
        "m4a": "audio_use/positive/5-pos.m4a",
        "wav": "audio_use/positive/5-pos.wav"
    },

    "id_006_pos": {
        "m4a": "audio_use/positive/6-pos.m4a",
        "wav": "audio_use/positive/6-pos.wav"
    },
    "id_007_pos": {
        "m4a": "audio_use/positive/7-pos.m4a",
        "wav": "audio_use/positive/7-pos.wav"
    },
    "id_008_pos": {
        "m4a": "audio_use/positive/8-pos.m4a",
        "wav": "audio_use/positive/8-pos.wav"
    },
    "id_009_pos": {
        "m4a": "audio_use/positive/9-pos.m4a",
        "wav": "audio_use/positive/9-pos.wav"
    },
    "id_010_pos":{
        "m4a": "audio_use/positive/10-pos.m4a",
        "wav": "audio_use/positive/10-pos.wav"
    },
    "id_011_pos":{
        "m4a": "audio_use/positive/11-pos.m4a",
        "wav": "audio_use/positive/11-pos.wav"
    },
    "id_012_pos":{
        "m4a": "audio_use/positive/12-pos.m4a",
        "wav": "audio_use/positive/12-pos.wav"
    },
    "id_013_pos":{
        "m4a": "audio_use/positive/13-pos.m4a",
        "wav": "audio_use/positive/13-pos.wav"
    },
    "id_014_pos":{
        "m4a": "audio_use/positive/14-pos.m4a",
        "wav": "audio_use/positive/14-pos.wav"
    },
    "id_015_pos":{
        "m4a": "audio_use/positive/15-pos.m4a",
        "wav": "audio_use/positive/15-pos.wav"
    },
    "id_016_pos":{
        "m4a": "audio_use/positive/16-pos.m4a",
        "wav": "audio_use/positive/16-pos.wav"
    },
}