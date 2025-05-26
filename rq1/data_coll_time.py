import os
import pandas as pd

# Example data (from user)
data = [
    {"anger":0.2332,"fear":0.1590,"joy":0.4244,"sadness":0.1214,"surprise":0.0620,"time":1.47},
    {"anger":0.1469,"fear":0.0342,"joy":0.6693,"sadness":0.0110,"surprise":0.1387,"time":5.15},
    {"anger":0.0993,"fear":0.0259,"joy":0.7804,"sadness":0.0184,"surprise":0.0759,"time":8.27},
    # ... rest of segments ...
    {"anger":0.1216,"fear":0.0837,"joy":0.5861,"sadness":0.0500,"surprise":0.1586,"time":43.2342}
]

df_segments = pd.DataFrame(data)

# Save to Excel
export_dir = 'exports_rq1_2'
os.makedirs(export_dir, exist_ok=True)
file_path = os.path.join(export_dir, 'id_example_segments.xlsx')
df_segments.to_excel(file_path, index=False)

df_segments.head()

import os, json, collections

counts = []
for fn in os.listdir("new_rq1_run_V3"):
    if not fn.endswith(".json"): continue
    data = json.load(open(os.path.join("new_rq1_run_V3", fn)))
    scores = data["praat_scores"].values()
    top = max(scores)
    ties = sum(1 for v in scores if v == top)
    counts.append(ties)

freq = collections.Counter(counts)
total = len(counts)
for ties, n in sorted(freq.items()):
    print(f"{ties} top emotion(s): {n} recordings ({n/total:.0%})")
