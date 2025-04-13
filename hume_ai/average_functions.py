#average_functions.py 

def compute_average_emotions(emotion_data: list[dict]) -> dict:
    if not emotion_data:
        return {}

    # Initialize sums
    totals = {}
    count = len(emotion_data)

    # Sum all values
    for entry in emotion_data:
        for emotion, score in entry.items():
            if emotion not in totals:
                totals[emotion] = 0.0
            totals[emotion] += score

    # Compute average per emotion
    averages = {emotion: total / count for emotion, total in totals.items()}
    return averages

