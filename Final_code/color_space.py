import numpy as np
def rgb_to_ycbcr(image):
    conversion_matrix = np.array([[ 0.299,     0.587,     0.114],
                                  [-0.168736, -0.331264,  0.5],
                                  [0.5,      -0.418688, -0.081312]])
    shift = np.array([0, 128, 128])
    ycbcr = image.dot(conversion_matrix.T) + shift
    return ycbcr.astype(np.float32)

def ycbcr_to_rgb(ycbcr):
    conversion_matrix = np.array([[1.0, 0.0, 1.402],
                                  [1.0, -0.344136, -0.714136],
                                  [1.0, 1.772, 0.0]])
    shift = np.array([0, -128, -128])
    rgb = (ycbcr + shift).dot(conversion_matrix.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)