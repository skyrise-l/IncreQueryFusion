from typing import Sequence
import numpy as np
import math
COS_EPS = 1e-9

@staticmethod 
def cosine(x: np.ndarray, y: np.ndarray) -> float:
    """余弦相似度（全局 ε 防止除零）"""
    num = float(np.dot(x, y))
    den = (np.linalg.norm(x) * np.linalg.norm(y)) + COS_EPS
    return num / den

@staticmethod 
def mean_vector(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """数值稳定的均值质心"""
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 1:                 # 只给了一条向量
            return vectors.copy()
        if vectors.size == 0:
            raise ValueError("empty vector array")
        return vectors.mean(axis=0)

    # 传进来的是 Sequence
    if len(vectors) == 0:
        raise ValueError("empty vector list")

    stacked = np.stack(vectors, axis=0)       # (n, d)
    return stacked.mean(axis=0)

@staticmethod 
def pairwise_cosine_avg(vectors: Sequence[np.ndarray]) -> float:
    """组内平均两两余弦，用于 similarity"""
    if len(vectors) == 1:
        return 1.0
    sims, n = 0.0, len(vectors)
    for i in range(n):
        sims += np.dot(vectors[i], vectors[(i + 1) % n]) 
    return sims / n

@staticmethod 
def is_null(val) -> bool:
    """空值判定：None / nan / 空字符串"""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        # 对于数字类型，检查是否为 NaN
        return math.isnan(val)
    except (TypeError, ValueError):
        # 如果 val 不是数字，math.isnan 会抛出 TypeError
        return False