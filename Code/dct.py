# dct.py
import numpy as np

def dct_2d(block):
    """
    Perform a 2D Discrete Cosine Transform on an 8x8 block.
    """
    block = block - 128  # Level shift
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for v in range(N):
            sum = 0.0
            for x in range(N):
                for y in range(N):
                    sum += block[x, y] * \
                           np.cos(((2*x +1) * u * np.pi) / 16) * \
                           np.cos(((2*y +1) * v * np.pi) / 16)
            alpha_u = 1/np.sqrt(2) if u == 0 else 1
            alpha_v = 1/np.sqrt(2) if v == 0 else 1
            result[u, v] = 0.25 * alpha_u * alpha_v * sum
    return result

def idct_2d(block):
    """
    Perform an inverse 2D Discrete Cosine Transform on an 8x8 block.
    """
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for x in range(N):
        for y in range(N):
            sum = 0.0
            for u in range(N):
                for v in range(N):
                    alpha_u = 1/np.sqrt(2) if u == 0 else 1
                    alpha_v = 1/np.sqrt(2) if v == 0 else 1
                    sum += alpha_u * alpha_v * block[u, v] * \
                           np.cos(((2*x +1) * u * np.pi) / 16) * \
                           np.cos(((2*y +1) * v * np.pi) / 16)
            result[x, y] = 0.25 * sum
    result = result + 128  # Reverse level shift
    return result
