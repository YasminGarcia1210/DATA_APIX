import numpy as np
from numpy.linalg import norm

def similitud_coseno(vec1, vec2):
    cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cos_sim

def es_match(cos_sim, umbral=0.35):
    # Cuanto más alto el cos_sim, más parecidos
    # 1.0 = idénticos
    return cos_sim >= (1 - umbral)
