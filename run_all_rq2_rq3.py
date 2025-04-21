import subprocess

print("Running Hume emotion analysis...")
subprocess.run(["python3", "hume_ai/run_emotion_analysis.py"], check=True)

print("Running full NLP + Hume + self-assessed analysis...")
subprocess.run(["python3", "full_analysis_rq2_rq3.py"], check=True)
