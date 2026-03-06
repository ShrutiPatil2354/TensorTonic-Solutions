import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):

    x1 = np.array(x1)
    x2 = np.array(x2)

    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    if norm_x1 == 0 or norm_x2 == 0:
        return 0.0

    cos = np.dot(x1, x2) / (norm_x1 * norm_x2)

    if label == 1:
        return float(1 - cos)

    if label == -1:
        return float(max(0, cos - margin))