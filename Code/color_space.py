# color_space.py
import numpy as np

def rgb_to_ycbcr(image):
    """
    Convert an RGB image to YCbCr color space.
    """
    # Define conversion matrix
    conversion_matrix = np.array([[ 0.299,     0.587,     0.114],
                                  [-0.168736, -0.331264, 0.5],
                                  [0.5,      -0.418688, -0.081312]])
    shift = np.array([0, 128, 128])

    # Reshape image for matrix multiplication
    ycbcr = image.dot(conversion_matrix.T) + shift
    return ycbcr.astype(np.float32)

def ycbcr_to_rgb(ycbcr):
    """
    Convert a YCbCr image back to RGB color space.
    """
    conversion_matrix = np.array([[1.0, 0.0, 1.402],
                                  [1.0, -0.344136, -0.714136],
                                  [1.0, 1.772, 0.0]])
    shift = np.array([0, -128, -128])

    ycbcr_shifted = ycbcr + shift
    rgb = ycbcr_shifted.dot(conversion_matrix.T)
    rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8)
