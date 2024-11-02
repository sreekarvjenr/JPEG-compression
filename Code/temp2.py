# import cv2
# import numpy as np
# import heapq
# from collections import defaultdict

# # ====================== Color Space Conversion ======================

# def rgb_to_ycbcr(image):
#     conversion_matrix = np.array([[ 0.299,     0.587,     0.114],
#                                   [-0.168736, -0.331264,  0.5],
#                                   [0.5,      -0.418688, -0.081312]])
#     shift = np.array([0, 128, 128])
#     ycbcr = image.dot(conversion_matrix.T) + shift
#     print("Checkpoint: Converted RGB to YCbCr color space")
#     return ycbcr.astype(np.float32)

# def ycbcr_to_rgb(ycbcr):
#     conversion_matrix = np.array([[1.0, 0.0, 1.402],
#                                   [1.0, -0.344136, -0.714136],
#                                   [1.0, 1.772, 0.0]])
#     shift = np.array([0, -128, -128])
#     rgb = (ycbcr + shift).dot(conversion_matrix.T)
#     print("Checkpoint: Converted YCbCr back to RGB color space")
#     return np.clip(rgb, 0, 255).astype(np.uint8)

# # ====================== DCT and IDCT ======================

# def dct_2d(block):
#     block = block - 128
#     N = 8
#     result = np.zeros((N, N), dtype=np.float32)
#     for u in range(N):
#         for v in range(N):
#             sum_val = 0.0
#             for x in range(N):
#                 for y in range(N):
#                     sum_val += block[x, y] * \
#                                np.cos(((2 * x + 1) * u * np.pi) / 16) * \
#                                np.cos(((2 * y + 1) * v * np.pi) / 16)
#             alpha_u = 1 / np.sqrt(2) if u == 0 else 1
#             alpha_v = 1 / np.sqrt(2) if v == 0 else 1
#             result[u, v] = 0.25 * alpha_u * alpha_v * sum_val
#     return result

# def idct_2d(block):
#     N = 8
#     result = np.zeros((N, N), dtype=np.float32)
#     for x in range(N):
#         for y in range(N):
#             sum_val = 0.0
#             for u in range(N):
#                 for v in range(N):
#                     alpha_u = 1 / np.sqrt(2) if u == 0 else 1
#                     alpha_v = 1 / np.sqrt(2) if v == 0 else 1
#                     sum_val += alpha_u * alpha_v * block[u, v] * \
#                                np.cos(((2 * x + 1) * u * np.pi) / 16) * \
#                                np.cos(((2 * y + 1) * v * np.pi) / 16)
#             result[x, y] = 0.25 * sum_val
#     return result + 128

# # ====================== Quantization ======================

# STANDARD_LUMINANCE_Q = np.array([
#     [16, 11, 10, 16, 24, 40, 51, 61],
#     [12, 12, 14, 19, 26, 58, 60, 55],
#     [14, 13, 16, 24, 40, 57, 69, 56],
#     [14, 17, 22, 29, 51, 87, 80, 62],
#     [18, 22, 37, 56, 68, 109, 103, 77],
#     [24, 35, 55, 64, 81, 104, 113, 92],
#     [49, 64, 78, 87, 103, 121, 120, 101],
#     [72, 92, 95, 98, 112, 100, 103, 99]
# ])

# STANDARD_CHROMINANCE_Q = np.array([
#     [17, 18, 24, 47, 99, 99, 99, 99],
#     [18, 21, 26, 66, 99, 99, 99, 99],
#     [24, 26, 56, 99, 99, 99, 99, 99],
#     [47, 66, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99],
#     [99, 99, 99, 99, 99, 99, 99, 99]
# ])

# def quantize(block, quant_table):
#     result = np.round(block / quant_table).astype(int)
#     print("Checkpoint: Quantized DCT block")
#     return result

# def dequantize(block, quant_table):
#     result = (block * quant_table).astype(np.float32)
#     print("Checkpoint: Dequantized block")
#     return result

# # ====================== Zigzag and Inverse Zigzag ======================

# def zigzag_order(block):
#     zigzag_indices = [
#         (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
#         (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
#         (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
#         (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
#         (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
#         (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
#         (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
#         (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
#     ]
#     return [block[i, j] for i, j in zigzag_indices]

# def inverse_zigzag_order(coefficients):
#     block = np.zeros((8, 8), dtype=np.float32)
#     zigzag_indices = [
#         (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
#         (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
#         (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
#         (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
#         (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
#         (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
#         (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
#         (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
#     ]
#     for idx, (i, j) in enumerate(zigzag_indices):
#         block[i, j] = coefficients[idx]
#     print("Checkpoint: Reconstructed block from Zigzag order")
#     return block

# # ====================== Huffman Coding ======================

# class HuffmanNode:
#     def __init__(self, symbol=None, freq=0):
#         self.symbol = symbol
#         self.freq = freq
#         self.left = None
#         self.right = None

#     def __lt__(self, other):
#         return self.freq < other.freq

# def build_huffman_tree(freq_table):
#     heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
#     heapq.heapify(heap)
    
#     while len(heap) > 1:
#         node1 = heapq.heappop(heap)
#         node2 = heapq.heappop(heap)
#         merged = HuffmanNode(None, node1.freq + node2.freq)
#         merged.left = node1
#         merged.right = node2
#         heapq.heappush(heap, merged)
    
#     return heap[0] if heap else None

# def build_codes(node, prefix="", codebook=None):
#     if codebook is None:
#         codebook = {}
#     if node is None:
#         return codebook
#     if node.symbol is not None:
#         codebook[node.symbol] = prefix
#     build_codes(node.left, prefix + "0", codebook)
#     build_codes(node.right, prefix + "1", codebook)
#     return codebook


# def huffman_encode(data):
#     freq_table = defaultdict(int)
#     for symbol in data:
#         freq_table[symbol] += 1
    
#     root = build_huffman_tree(freq_table)
#     codes = build_codes(root)
#     encoded_data = ''.join(codes[symbol] for symbol in data)
#     print("Checkpoint: Huffman encoded data")
#     return encoded_data, codes

# def huffman_decode(encoded_data, codes):
#     reversed_codes = {v: k for k, v in codes.items()}
#     current_code = ""
#     decoded_data = []
    
#     for bit in encoded_data:
#         current_code += bit
#         if current_code in reversed_codes:
#             decoded_data.append(reversed_codes[current_code])
#             current_code = ""
#     print("Checkpoint: Huffman decoded data")
#     return decoded_data

# # ====================== Run-Length Encoding and Decoding ======================

# def run_length_encode(block):
#     rle = []
#     zero_count = 0
#     dc = block[0]
#     ac = block[1:]
#     for coeff in ac:
#         if coeff == 0:
#             zero_count += 1
#         else:
#             while zero_count > 15:
#                 rle.append(('AC', 15, 0))
#                 zero_count -= 16
#             rle.append(('AC', zero_count, coeff))
#             zero_count = 0
#     if zero_count > 0:
#         rle.append(('EOB',))
#     print("Checkpoint: Run-length encoded AC coefficients")
#     return ('DC', dc), rle


# def run_length_decode(dc_symbol, rle):
#     dc = dc_symbol[1]
#     coefficients = [dc]
#     for item in rle:
#         zeros, coeff = item
#         coefficients.extend([0] * zeros)
#         if coeff != 0:
#             coefficients.append(coeff)
#     coefficients = coefficients[:64] + [0] * (64 - len(coefficients))
#     print("Checkpoint: Decoded run-length encoded coefficients")
#     return coefficients


# # ====================== Image Padding and Blocking ======================

# def pad_image(image, block_size=8):
#     h, w, c = image.shape
#     pad_h = (block_size - (h % block_size)) if (h % block_size) != 0 else 0
#     pad_w = (block_size - (w % block_size)) if (w % block_size) != 0 else 0
#     padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
#     return padded_image

# def block_split(channel, block_size=8):
#     h, w = channel.shape
#     blocks = []
#     for i in range(0, h, block_size):
#         for j in range(0, w, block_size):
#             block = channel[i:i+block_size, j:j+block_size]
#             blocks.append(block)
#     return blocks

# def block_merge(blocks, height, width, block_size=8):
#     channel = np.zeros((height, width), dtype=np.float32)
#     blocks_per_row = width // block_size
#     for idx, block in enumerate(blocks):
#         i = (idx // blocks_per_row) * block_size
#         j = (idx % blocks_per_row) * block_size
#         channel[i:i+block_size, j:j+block_size] = block
#     return channel

# # ====================== Channel Processing ======================

# def process_channel(channel, quant_table):
#     blocks = block_split(channel)
#     print(f"Checkpoint: Split channel into {len(blocks)} blocks")
    
#     dct_blocks = [dct_2d(block) for block in blocks]
#     print("Checkpoint: Applied DCT to each block")

#     quantized_blocks = [quantize(block, quant_table) for block in dct_blocks]
#     print("Checkpoint: Quantized each block")

#     zigzagged = [zigzag_order(block) for block in quantized_blocks]
#     print("Checkpoint: Zigzagged each block")

#     flat_coeffs = []
#     for block in zigzagged:
#         dc_symbol, rle = run_length_encode(block)
#         flat_coeffs.append(dc_symbol)
#         flat_coeffs.extend(rle)
#     print("Checkpoint: Run-length encoded each block")

#     # Now Huffman encode the flat_coeffs
#     encoded_data, codes = huffman_encode(flat_coeffs)
#     return encoded_data, codes


# def inverse_process_channel(encoded_data, codes, quant_table, height, width):
#     # Huffman decode the data
#     decoded_data = huffman_decode(encoded_data, codes)
#     decoded_blocks = []
#     idx = 0

#     while idx < len(decoded_data):
#         symbol = decoded_data[idx]
#         idx += 1

#         if symbol[0] == 'DC':
#             dc = symbol[1]
#             rle = []
#             while idx < len(decoded_data):
#                 symbol = decoded_data[idx]
#                 idx += 1
#                 if symbol == ('EOB',):
#                     break
#                 elif symbol[0] == 'AC':
#                     zeros, coeff = symbol[1], symbol[2]
#                     rle.append((zeros, coeff))
#             coeffs = run_length_decode(dc, rle)
#             block = inverse_zigzag_order(coeffs)
#             dequant = dequantize(block, quant_table)
#             idct_block = idct_2d(dequant)
#             decoded_blocks.append(idct_block)
#         else:
#             raise ValueError("Unexpected symbol in decoded data")
#     print("Checkpoint: Inverse processed each block to reconstruct channel")

#     # Merge blocks and ensure dimensions match the padded size
#     padded_height = (height + 7) // 8 * 8
#     padded_width = (width + 7) // 8 * 8
#     channel = block_merge(decoded_blocks, padded_height, padded_width)
    
#     # Crop to the original dimensions
#     channel = channel[:height, :width]
#     return channel


# # ====================== JPEG Compression and Decompression ======================

# def jpeg_compress(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or unable to read.")
#     ycbcr = rgb_to_ycbcr(image)
#     padded = pad_image(ycbcr)
#     height, width, _ = ycbcr.shape
#     Y, Cb, Cr = padded[:,:,0], padded[:,:,1], padded[:,:,2]
    
#     Y_data, Y_codes = process_channel(Y, STANDARD_LUMINANCE_Q)
#     Cb_data, Cb_codes = process_channel(Cb, STANDARD_CHROMINANCE_Q)
#     Cr_data, Cr_codes = process_channel(Cr, STANDARD_CHROMINANCE_Q)
#     print("Checkpoint: Processed Y, Cb, and Cr channels")

#     return {
#         'Y': (Y_data, Y_codes),
#         'Cb': (Cb_data, Cb_codes),
#         'Cr': (Cr_data, Cr_codes),
#         'height': height,
#         'width': width
#     }

# def jpeg_decompress(compressed):
#     (Y_data, Y_codes) = compressed['Y']
#     (Cb_data, Cb_codes) = compressed['Cb']
#     (Cr_data, Cr_codes) = compressed['Cr']
#     height, width = compressed['height'], compressed['width']
    
#     Y = inverse_process_channel(Y_data, Y_codes, STANDARD_LUMINANCE_Q, height, width)
#     Cb = inverse_process_channel(Cb_data, Cb_codes, STANDARD_CHROMINANCE_Q, height, width)
#     Cr = inverse_process_channel(Cr_data, Cr_codes, STANDARD_CHROMINANCE_Q, height, width)
    
#     ycbcr = np.stack((Y, Cb, Cr), axis=2)
#     rgb = ycbcr_to_rgb(ycbcr)
#     print("Checkpoint: Reconstructed image from Y, Cb, Cr channels")
#     return rgb

# # ====================== Main ======================

# # ... [Other functions remain the same with the above modifications applied]

# # ====================== Main ======================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='JPEG Compression Pipeline')
#     parser.add_argument('input_image', type=str, help='Path to input image')
#     parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
#     args = parser.parse_args()

#     compressed = jpeg_compress(args.input_image)
#     reconstructed = jpeg_decompress(compressed)
#     cv2.imwrite(args.output_image, reconstructed)
#     print(f"Reconstructed image saved to {args.output_image}")

import cv2
import numpy as np
import heapq
from collections import defaultdict

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
    if N == 0:
        raise ValueError("Block size cannot be zero")
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
    if N == 0:
        raise ValueError("Block size cannot be zero")
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
    try:
        for idx, (i, j) in enumerate(zigzag_indices):
            block[i, j] = coefficients[idx]
    except IndexError:
        print("Warning: Coefficients list is shorter than expected. Padding with zeros.")
        # Calculate the number of remaining coefficients needed
        remaining_coefficients = 64 - len(coefficients)
        # Pad the coefficients list with zeros
        coefficients += [0] * remaining_coefficients
        # Retry assigning coefficients to the block
        for idx, (i, j) in enumerate(zigzag_indices):
            block[i, j] = coefficients[idx]
    print("Checkpoint: Reconstructed block from Zigzag order")
    return block
# ====================== Huffman Coding ======================
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_table):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_table.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def build_codes(node, prefix="", codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data):
    freq_table = defaultdict(int)
    for symbol in data:
        freq_table[symbol] += 1
    
    root = build_huffman_tree(freq_table)
    codes = build_codes(root)
    encoded_data = ''.join(codes[symbol] for symbol in data)
    print("Checkpoint: Huffman encoded data")
    return encoded_data, codes

def huffman_decode(encoded_data, codes):
    reversed_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_codes:
            decoded_data.append(reversed_codes[current_code])
            current_code = ""
    print("Checkpoint: Huffman decoded data")
    return decoded_data

def run_length_encode(ac_coefficients):
    rle = []
    zero_count = 0
    for coeff in ac_coefficients:
        if coeff == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                rle.append(('AC', 15, 0))
                zero_count -= 16
            rle.append(('AC', zero_count, coeff))
            zero_count = 0
    if zero_count > 0:
        rle.append(('AC', 0, 0))  # End of Block
    return rle

# ====================== JPEG Compression and Decompression ======================
def process_block(block, quant_table):
    dct_block = dct_2d(block)
    quantized_block = quantize(dct_block, quant_table)
    zigzagged_block = zigzag_order(quantized_block)
    dc_coefficient = zigzagged_block[0]
    ac_coefficients = zigzagged_block[1:]
    rle_ac = run_length_encode(ac_coefficients)
    # Combine DC and AC coefficients
    symbols = [('DC', dc_coefficient)] + rle_ac + [('EOB',)]
    encoded_block, codes = huffman_encode(symbols)
    return encoded_block, codes



def pad_image(image, block_size=8):
    h, w, c = image.shape
    pad_h = (block_size - (h % block_size)) if (h % block_size)!= 0 else 0
    pad_w = (block_size - (w % block_size)) if (w % block_size)!= 0 else 0
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0,0)), mode='constant', constant_values=0)
    return padded_image

def block_split(channel, block_size=8):
    h, w = channel.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return blocks

def block_merge(blocks, height, width, block_size=8):
    channel = np.zeros((height, width), dtype=np.float32)
    blocks_per_row = width // block_size
    for idx, block in enumerate(blocks):
        i = (idx // blocks_per_row) * block_size
        j = (idx % blocks_per_row) * block_size
        channel[i:i+block_size, j:j+block_size] = block
    return channel

def jpeg_compress(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    ycbcr = rgb_to_ycbcr(image)
    padded = pad_image(ycbcr)
    height, width, _ = ycbcr.shape
    
    Y, Cb, Cr = padded[:,:,0], padded[:,:,1], padded[:,:,2]
    Y_blocks = block_split(Y)
    Cb_blocks = block_split(Cb)
    Cr_blocks = block_split(Cr)
    
    Y_compressed = []
    Y_codes_list = []
    for block in Y_blocks:
        encoded_block, codes = process_block(block, STANDARD_LUMINANCE_Q)
        Y_compressed.extend(encoded_block)
        Y_codes_list.append(codes)
    
    Cb_compressed = []
    Cb_codes_list = []
    for block in Cb_blocks:
        encoded_block, codes = process_block(block, STANDARD_CHROMINANCE_Q)
        Cb_compressed.extend(encoded_block)
        Cb_codes_list.append(codes)
    
    Cr_compressed = []
    Cr_codes_list = []
    for block in Cr_blocks:
        encoded_block, codes = process_block(block, STANDARD_CHROMINANCE_Q)
        Cr_compressed.extend(encoded_block)
        Cr_codes_list.append(codes)
    
    return {
        'Y': (Y_compressed, Y_codes_list),
        'Cb': (Cb_compressed, Cb_codes_list),
        'Cr': (Cr_compressed, Cr_codes_list),
        'height': height,
        'width': width
    }

def run_length_decode(rle_block):
    coefficients = []
    i = 0
    while i < len(rle_block):
        if isinstance(rle_block[i], tuple) and rle_block[i]!= (0, 0):
            coefficients.extend([0] * rle_block[i][0])
            coefficients.append(rle_block[i][1])
        elif rle_block[i]!= (0, 0):
            coefficients.append(rle_block[i])
        i += 1
        if i < len(rle_block) and rle_block[i] == (0, 0):
            break
    # Pad with zeros to fill the 64 coefficients
    coefficients = coefficients[:64] + [0] * (64 - len(coefficients))
    return coefficients

def inverse_process_block(encoded_block, codes, quant_table):
    try:
        decoded_block = huffman_decode(encoded_block, codes)
        # Handle RLE decoding
        rle_blocks = []
        temp_block = []
        for coeff in decoded_block:
            if isinstance(coeff, tuple) and coeff == (0, 0):
                rle_blocks.append(temp_block)
                temp_block = []
            else:
                temp_block.append(coeff)
        if temp_block:
            rle_blocks.append(temp_block)
        
        reconstructed_blocks = []
        for rle_block in rle_blocks:
            coefficients = run_length_decode(rle_block)
            block = inverse_zigzag_order(coefficients)
            dequantized_block = dequantize(block, quant_table)
            idct_block = idct_2d(dequantized_block)
            reconstructed_blocks.append(idct_block)
        return reconstructed_blocks
    except Exception as e:
        print(f"Error in inverse_process_block: {str(e)}")
        return []  # or return a default value/block

def jpeg_decompress(compressed):
    try:
        Y_compressed, Y_codes_list = compressed['Y']
        Cb_compressed, Cb_codes_list = compressed['Cb']
        Cr_compressed, Cr_codes_list = compressed['Cr']
        height, width = compressed['height'], compressed['width']
        
        Y_reconstructed_blocks = []
        for i, block in enumerate(Y_compressed):
            try:
                reconstructed_blocks = inverse_process_block(block, Y_codes_list[i], STANDARD_LUMINANCE_Q)
                Y_reconstructed_blocks.extend(reconstructed_blocks)
            except Exception as e:
                print(f"Error in Y block {i}: {str(e)}")
        
        Cb_reconstructed_blocks = []
        for i, block in enumerate(Cb_compressed):
            try:
                reconstructed_blocks = inverse_process_block(block, Cb_codes_list[i], STANDARD_CHROMINANCE_Q)
                Cb_reconstructed_blocks.extend(reconstructed_blocks)
            except Exception as e:
                print(f"Error in Cb block {i}: {str(e)}")
        
        Cr_reconstructed_blocks = []
        for i, block in enumerate(Cr_compressed):
            try:
                reconstructed_blocks = inverse_process_block(block, Cr_codes_list[i], STANDARD_CHROMINANCE_Q)
                Cr_reconstructed_blocks.extend(reconstructed_blocks)
            except Exception as e:
                print(f"Error in Cr block {i}: {str(e)}")
        
        # Merge the reconstructed blocks into channels
        Y_channel = block_merge(Y_reconstructed_blocks, height, width)
        Cb_channel = block_merge(Cb_reconstructed_blocks, height, width)
        Cr_channel = block_merge(Cr_reconstructed_blocks, height, width)
        
        ycbcr = np.stack((Y_channel, Cb_channel, Cr_channel), axis=2)
        rgb = ycbcr_to_rgb(ycbcr)
        print("Checkpoint: Reconstructed image from Y, Cb, Cr channels")
        return rgb
    except Exception as e:
        print(f"Error in jpeg_decompress: {str(e)}")
        return None

# ====================== Main ======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='JPEG Compression Pipeline')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
    args = parser.parse_args()

    try:
        compressed = jpeg_compress(args.input_image)
        reconstructed = jpeg_decompress(compressed)
        if reconstructed is not None:
            cv2.imwrite(args.output_image, reconstructed)
            print(f"Reconstructed image saved to {args.output_image}")
        else:
            print("Failed to reconstruct image.")
    except Exception as e:
        print(f"Error: {str(e)}")