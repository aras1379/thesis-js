# analyze_emotion.py
import requests
import json
import os
import re 
import nlpcloud 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NLP_API_KEY")
client = nlpcloud.Client("finetuned-llama-3-70b", api_key, gpu=True)

def emotion_analyze(text: str):
    prompt = (
        "Analyze the emotions in the following Swedish text. "
        "Return a JSON dictionary with the emotion labels as keys and scores between 0.0 and 1.0 as values. "
        "Use these labels only: joy, surprise, fear, anger, disgust, sadness.\n\n"
        f"Text: {text}"
    )
    
    response = client.generation(
        prompt, 
        max_length = 300,
        temperature = 0.5, 
        top_p=0.9
    )
    
    generated_text = response.get("generated_text", "").strip() 
    
    match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            emotion_scores = json.loads(json_str)
        except json.JSONDecodeError:
            print("⚠️ JSON block found but failed to parse:")
            print(json_str)
            emotion_scores = {}
    else:
        print("⚠️ No JSON block found in model output:")
        print(generated_text)
        emotion_scores = {}

    return emotion_scores