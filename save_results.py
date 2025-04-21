#save_results.py 

import os 
import json 

def save_combined_result(entry_id, result_data, output_path="results_combined_rq2_rq3.json"):
    # Load existing results if file exists
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Add/update the current entry
    all_results[entry_id] = result_data

    # Save back to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f" Appended/updated entry '{entry_id}' in '{output_path}'")