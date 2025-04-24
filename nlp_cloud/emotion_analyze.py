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
def safe_extract_json(text):
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = json_str.rstrip(", \n") + "}"
            try:
                return json.loads(json_str)
            except:
                print("Couldn't parse:")
                print(json_str)
    else:
        print("No JSON block found in model output:")
        print(text)
    return {}

def emotion_analyze(text: str):
    prompt = (
        "Du är ett avancerat system för emotionsanalys av svenska texter.\n\n"
        "Givet följande text, GE ENDAST ett JSON-objekt med EXAKT dessa fem nycklar:\n"
        "  anger, joy, sadness, fear, surprise\n\n"
        "Varje värde ska vara ett decimaltal mellan 0.0 och 1.0 med tre decimaler (t.ex. 0.123).\n"
        "Du får inte skriva något annat – bara det rena JSON-objektet.\n\n"
        f"TEXT:\n{text}"
    )

    response = client.generation(
        prompt, 
        max_length=1024,
        temperature=0.3,
        top_p=0.9
    )

    generated_text = response.get("generated_text", "").strip()
    return safe_extract_json(generated_text)