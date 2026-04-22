import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute the cosine similarity between two vectors. Supports input as lists or NumPy arrays.

    Args:
        vec1 (list[float] or np.ndarray): The first vector.
        vec2 (list[float] or np.ndarray): The second vector.

    Returns:
        float: Cosine similarity, ranging from -1 to 1.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Only one-dimensional vectors are supported.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0 

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [1, 2, 3]
    print(cosine_similarity(vec1, vec2))