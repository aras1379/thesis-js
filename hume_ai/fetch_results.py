#fetch_results.py

import os
import requests
from dotenv import load_dotenv

#loading api
load_dotenv()
api_key = os.getenv("HUME_API_KEY")

HUME_RESULTS_URL = "https://api.hume.ai/v0/batch/jobs/{job_id}/predictions"


def get_analysis_results(job_id):
    # Fetches result from Hume AI for the specific job ID

    headers = {
        "X-Hume-Api-Key": api_key
    }
    response = requests.get(HUME_RESULTS_URL.format(job_id=job_id), headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error fetching results: {response.status_code} {response.text}")
    
    return response.json()