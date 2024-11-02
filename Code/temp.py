import cv2
import numpy as np

# ====================== Color Space Conversion ======================

def rgb_to_ycbcr(image):
    conversion_matrix = np.array([[ 0.299,     0.587,     0.114],
                                  [-0.168736, -0.331264,  0.5],
                                  [0.5,      -0.418688, -0.081312]])
    shift = np.array([0, 128, 128])
    ycbcr = image.dot(conversion_matrix.T) + shift
    print("Checkpoint: Converted RGB to YCbCr color space")
    return ycbcr.astype(np.float32)

def ycbcr_to_rgb(ycbcr):
    conversion_matrix = np.array([[1.0, 0.0, 1.402],
                                  [1.0, -0.344136, -0.714136],
                                  [1.0, 1.772, 0.0]])
    shift = np.array([0, -128, -128])
    rgb = (ycbcr + shift).dot(conversion_matrix.T)
    print("Checkpoint: Converted YCbCr back to RGB color space")
    return np.clip(rgb, 0, 255).astype(np.uint8)

# ====================== DCT and IDCT ======================

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

# ====================== Quantization ======================

STANDARD_LUMINANCE_Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

STANDARD_CHROMINANCE_Q = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def quantize(block, quant_table):
    result = np.round(block / quant_table).astype(int)
    print("Checkpoint: Quantized DCT block")
    return result

def dequantize(block, quant_table):
    result = (block * quant_table).astype(np.float32)
    print("Checkpoint: Dequantized block")
    return result

# ====================== Zigzag and Inverse Zigzag ======================

def zigzag_order(block):
    zigzag_indices = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    return [block[i, j] for i, j in zigzag_indices]

def inverse_zigzag_order(coefficients):
    block = np.zeros((8, 8), dtype=np.float32)
    zigzag_indices = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    for idx, (i, j) in enumerate(zigzag_indices):
        block[i, j] = coefficients[idx]
    print("Checkpoint: Reconstructed block from Zigzag order")
    return block

# ====================== Run-Length Encoding (RLE) ======================

def run_length_encode(block):
    rle = []
    zero_count = 0
    dc = block[0]
    ac = block[1:]
    for coeff in ac:
        if coeff == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                rle.append((15, 0))
                zero_count -= 16
            rle.append((zero_count, coeff))
            zero_count = 0
    if zero_count > 0:
        rle.append((0, 0))  # End of Block
    print("Checkpoint: Run-length encoded AC coefficients")
    return (dc, rle)

def run_length_decode(dc, rle):
    coefficients = [dc]
    for item in rle:
        if isinstance(item, tuple):
            zeros, coeff = item
            coefficients.extend([0] * zeros)
            if coeff != 0:
                coefficients.append(coeff)
    coefficients = coefficients[:64] + [0] * (64 - len(coefficients))
    print("Checkpoint: Decoded run-length encoded coefficients")
    return coefficients

# ====================== Channel Processing Functions ======================
def pad_image(image, block_size=8):
    """
    Pads the image so that its dimensions are multiples of block_size.
    """
    h, w, c = image.shape
    pad_h = (block_size - (h % block_size)) if (h % block_size) != 0 else 0
    pad_w = (block_size - (w % block_size)) if (w % block_size) != 0 else 0
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
    return padded_image

def block_split(channel, block_size=8):
    """
    Splits a single channel into non-overlapping block_size x block_size blocks.
    """
    h, w = channel.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def block_merge(blocks, height, width, block_size=8):
    """
    Merges 8x8 blocks back into a single channel.
    """
    channel = np.zeros((height, width), dtype=np.float32)
    blocks_per_row = width // block_size
    for idx, block in enumerate(blocks):
        i = (idx // blocks_per_row) * block_size
        j = (idx % blocks_per_row) * block_size
        channel[i:i+block_size, j:j+block_size] = block
    return channel
import cv2
import numpy as np

# ... (Other functions remain the same)

# ====================== Channel Processing Functions ======================

def process_channel(channel, quant_table):
    blocks = block_split(channel)
    print(f"Checkpoint: Split channel into {len(blocks)} blocks")
    
    dct_blocks = [dct_2d(block) for block in blocks]
    print("Checkpoint: Applied DCT to each block")

    quantized_blocks = [quantize(block, quant_table) for block in dct_blocks]
    print("Checkpoint: Quantized each block")

    zigzagged = [zigzag_order(block) for block in quantized_blocks]
    print("Checkpoint: Zigzagged each block")

    flat_coeffs = []
    for block in zigzagged:
        dc, rle = run_length_encode(block)
        flat_coeffs.append(dc)
        flat_coeffs.extend(rle)
    print("Checkpoint: Run-length encoded each block")
    
    return flat_coeffs

def inverse_process_channel(encoded_data, quant_table, height, width):
    decoded_blocks = []
    idx = 0

    while idx < len(encoded_data):
        dc = encoded_data[idx]
        idx += 1
        rle = []

        while idx < len(encoded_data):
            symbol = encoded_data[idx]
            idx += 1
            if isinstance(symbol, tuple) and symbol == (0, 0):
                break
            rle.append(symbol)

        coeffs = run_length_decode(dc, rle)
        block = inverse_zigzag_order(coeffs)
        dequant = dequantize(block, quant_table)
        idct_block = idct_2d(dequant)
        decoded_blocks.append(idct_block)
    
    print("Checkpoint: Inverse processed each block to reconstruct channel")

    # Merge blocks and ensure dimensions match the padded size
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    channel = block_merge(decoded_blocks, padded_height, padded_width)
    
    # Crop to the original dimensions
    channel = channel[:height, :width]
    return channel

# ====================== JPEG Compression and Decompression ======================

def jpeg_compress(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    ycbcr = rgb_to_ycbcr(image)
    padded = pad_image(ycbcr)
    height, width, _ = ycbcr.shape
    Y, Cb, Cr = padded[:,:,0], padded[:,:,1], padded[:,:,2]
    
    Y_data = process_channel(Y, STANDARD_LUMINANCE_Q)
    Cb_data = process_channel(Cb, STANDARD_CHROMINANCE_Q)
    Cr_data = process_channel(Cr, STANDARD_CHROMINANCE_Q)
    print("Checkpoint: Processed Y, Cb, and Cr channels")

    return {'Y': Y_data, 'Cb': Cb_data, 'Cr': Cr_data, 'height': height, 'width': width}

def jpeg_decompress(compressed):
    Y = inverse_process_channel(compressed['Y'], STANDARD_LUMINANCE_Q, compressed['height'], compressed['width'])
    Cb = inverse_process_channel(compressed['Cb'], STANDARD_CHROMINANCE_Q, compressed['height'], compressed['width'])
    Cr = inverse_process_channel(compressed['Cr'], STANDARD_CHROMINANCE_Q, compressed['height'], compressed['width'])
    
    # Ensure Y, Cb, and Cr channels have the same dimensions
    assert Y.shape == Cb.shape == Cr.shape, "Y, Cb, and Cr channels have different dimensions after decompression"
    
    ycbcr = np.stack((Y, Cb, Cr), axis=2)
    rgb = ycbcr_to_rgb(ycbcr)
    print("Checkpoint: Reconstructed image from Y, Cb, Cr channels")
    return rgb

# ====================== Main ======================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='JPEG Compression Pipeline')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
    args = parser.parse_args()

    compressed = jpeg_compress(args.input_image)
    reconstructed = jpeg_decompress(compressed)
    cv2.imwrite(args.output_image, reconstructed)
    print(f"Reconstructed image saved to {args.output_image}")

