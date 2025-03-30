# analyze_emotion.py
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NLP_API_KEY")

def emotion_analyze(text: str):
    url = "https://api.nlpcloud.io/v1/gpu/finetuned-llama-3-70b/sentiment"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": f"Analyze the emotions in the following Swedish text and list the dominant emotion(s):\n\n{text}",
        "use_gpu": True,
        "max_length": 200,
        "temperature": 0.5,
        "top_p": 0.9
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise Exception(f"Emotion Analysis Error: {response.status_code} {response.text}")

    return response.json()["scored_labels"]