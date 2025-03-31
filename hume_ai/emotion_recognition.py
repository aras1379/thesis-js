# emotion_recognition.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HUME_API_KEY")

HUME_API_URL = "https://api.hume.ai/v0/batch/jobs"

def analyze_audio(file_path):
    # sends a local audio file to Hume AI for emotion analysis
    # vocal burst & speech prosody

    headers = { 
        "X-Hume-Api-Key": api_key
    }

    # models in use
    job_config = {
        "models": {
            "prosody": {},
            "burst": {}
        }
    }

    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(HUME_API_URL, headers=headers, files=files, json = job_config)

        # response handling
        if response.status_code !=200:
            raise Exception(f"Error could not start job: {response.status_code} {response.text}")
        
        job_id = response.json() ["job_id"]
        print(f"job started successfully! Job ID: {job_id}")
        return job_id