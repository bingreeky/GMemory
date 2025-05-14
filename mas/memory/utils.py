import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    计算两个向量之间的余弦相似度，支持列表或numpy数组作为输入。

    参数:
        vec1 (list[float] 或 np.ndarray): 向量1
        vec2 (list[float] 或 np.ndarray): 向量2

    返回:
        float: 余弦相似度，范围在 [-1, 1] 之间
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("只支持一维向量")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # 若某个向量全为0，定义相似度为0

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [1, 2, 3]
    print(cosine_similarity(vec1, vec2))