import numpy as np
from scipy.spatial.distance import cosine

LABEL_LIST = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def one_hot_vector(label, label_list=LABEL_LIST):
    return np.array([1.0 if e == label else 0.0 for e in label_list])

def hume_vector(hume_scores, label_list=LABEL_LIST):
    # Use normalized Hume scores and ensure order
    return np.array([hume_scores.get(e, 0.0) for e in label_list])

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)
