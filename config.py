#config.py 

#Change this to the current audio for analyzation 
active_audio_id = "id_001_pos"

emotions_to_analyze = [
    "anger", "joy", "sadness", "surprise (positive)", 
    "surprise (negative)", "fear", "disgust"
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
    "id_003_neg":{
        "m4a": "audio_use/negative/3-neg.m4a",
        "wav": "audio_use/negative/3-neg.wav"
    },
    "id_004_neg": {
        "m4a": "audio_use/negative/4-neg1.m4a",
        "wav": "audio_use/negative/4-neg1.wav"
    },
    "id_0042_neg": {
        "m4a": "audio_use/negative/4-neg2.m4a",
        "wav": "audio_use/negative/4-neg2.wav"
    },
    "id_0043_neg": {
        "m4a": "audio_use/negative/4-neg3.m4a",
        "wav": "audio_use/negative/4-neg3.wav"
    },
    "id_005_neg": {
        "m4a": "audio_use/negative/5-neg.m4a",
        "wav": "audio_use/negative/5-neg.wav"
    },
    "id_sara_neg": {
        "m4a": "audio_use/negative/sara-neg.m4a",
        "wav": "audio_use/negative/sara-neg.wav"
    },
    "id_006_neg": {
        "m4a": "audio_use/negative/6-neg.m4a",
        "wav": "audio_use/negative/6-neg.wav"
    },
    "id_007_neg": {
        "m4a": "audio_use/negative/7-neg.m4a",
        "wav": "audio_use/negative/7-neg.wav"
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
    "id_003_pos": {
        "m4a": "audio_use/positive/3-pos.m4a",
        "wav": "audio_use/positive/3-pos.wav"
    },
    "id_004_pos": {
        "m4a": "audio_use/positive/4-pos.m4a",
        "wav": "audio_use/positive/4-pos.wav"
    },
    "id_005_pos": {
        "m4a": "audio_use/positive/5-pos.m4a",
        "wav": "audio_use/positive/5-pos.wav"
    },
    "id_sara_pos": {
        "m4a": "audio_use/positive/sara-pos.m4a",
        "wav": "audio_use/positive/sara-pos.wav"
    },
    "id_006_pos": {
        "m4a": "audio_use/positive/6-pos.m4a",
        "wav": "audio_use/positive/6-pos.wav"
    },
    "id_007_pos": {
        "m4a": "audio_use/positive/7-pos.m4a",
        "wav": "audio_use/positive/7-pos.wav"
    }
}