# hume_utils.py
import json

def load_hume_average(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load Hume average: {e}")
        return {}
    