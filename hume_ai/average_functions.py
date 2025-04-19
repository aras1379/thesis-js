#average_functions.py 

def compute_average_emotions(emotion_data: list[dict]) -> dict:
    if not emotion_data:
        return {}

    totals = {}
    count = len(emotion_data)

    for entry in emotion_data:
        for emotion, score in entry.items():
            if emotion == "time":
                continue
            totals[emotion] = totals.get(emotion, 0.0) + score

    averages = {emotion: round(total / count, 3) for emotion, total in totals.items()}
    return averages

