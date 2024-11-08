import numpy as np

def dct_2d(block):
    block = block - 128
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for u in range(N):
        for v in range(N):
            sum_val = 0.0
            for x in range(N):
                for y in range(N):
                    sum_val += block[x, y] * \
                               np.cos(((2 * x + 1) * u * np.pi) / 16) * \
                               np.cos(((2 * y + 1) * v * np.pi) / 16)
            alpha_u = 1 / np.sqrt(2) if u == 0 else 1
            alpha_v = 1 / np.sqrt(2) if v == 0 else 1
            result[u, v] = 0.25 * alpha_u * alpha_v * sum_val
    return result

def idct_2d(block):
    N = 8
    result = np.zeros((N, N), dtype=np.float32)
    for x in range(N):
        for y in range(N):
            sum_val = 0.0
            for u in range(N):
                for v in range(N):
                    alpha_u = 1 / np.sqrt(2) if u == 0 else 1
                    alpha_v = 1 / np.sqrt(2) if v == 0 else 1
                    sum_val += alpha_u * alpha_v * block[u, v] * \
                               np.cos(((2 * x + 1) * u * np.pi) / 16) * \
                               np.cos(((2 * y + 1) * v * np.pi) / 16)
            result[x, y] = 0.25 * sum_val
    return result + 128