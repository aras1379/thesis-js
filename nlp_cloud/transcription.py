import base64
import requests
import json

from dotenv import load_dotenv 
import os 
from config import audio_files, active_audio_id

load_dotenv()
api_key = os.getenv("NLP_API_KEY")

entry_id = active_audio_id
audio_path = audio_files[entry_id]["wav"]

def transcribe_audio(filepath: str) -> str: 
    # Read audiofile and convert to base64 (to run locally)
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        base64_encoded = base64.b64encode(audio_data).decode('utf-8')
    
    url = "https://api.nlpcloud.io/v1/gpu/whisper/asr"
    
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "encoded_file": base64_encoded,
        "language": "sv"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code != 200:
        raise Exception(f"ASR Error: {response.status_code} {response.text}")
    
    return response.json()['text']



