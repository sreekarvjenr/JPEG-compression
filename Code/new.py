import cv2
import numpy as np
import heapq
from collections import defaultdict
import os
# ====================== Color Space Conversion ======================
def calculate_compression_ratio(original_image_path, compressed_data):
    # Get the size of the original image
    original_size = os.path.getsize(original_image_path)  # Size in bytes

    # Calculate the size of the compressed data in bits
    compressed_size_bits = 0
    for channel in ['Y', 'Cb', 'Cr']:
        for encoded_block, _ in compressed_data[channel]:
            compressed_size_bits += len(encoded_block)  # Each character is 1 bit

    # Convert compressed size from bits to bytes
    compressed_size = compressed_size_bits / 8

    # Calculate the compression ratio
    compression_ratio = original_size / compressed_size

    return compression_ratio
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
    return result

def dequantize(block, quant_table):
    result = (block * quant_table).astype(np.float32)
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
    if len(coefficients) < 64:
        coefficients.extend([0] * (64 - len(coefficients)))
    elif len(coefficients) > 64:
        coefficients = coefficients[:64]
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

def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.left is None and node.right is None:
        codeword = prefix if prefix != '' else '0'  # Assign '0' if prefix is empty
        codebook[node.symbol] = codeword
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data, log_file=None):
    freq_table = defaultdict(int)
    for symbol in data:
        freq_table[symbol] += 1
    
    root = build_huffman_tree(freq_table)
    codes = build_codes(root)
    # Log the Huffman codes
    if log_file:
        log_file.write("Huffman codes for symbols:\n")
        for symbol in codes:
            log_file.write(f"Symbol: {symbol}, Code: {codes[symbol]}\n")
    encoded_data = ''.join(codes[symbol] for symbol in data)
    return encoded_data, codes

def huffman_decode(encoded_data, codes, log_file=None):
    reversed_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_codes:
            decoded_data.append(reversed_codes[current_code])
            current_code = ""
    return decoded_data

# ====================== Run-Length Encoding and Decoding ======================

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

def run_length_decode(ac_rle):
    coefficients = []
    for zeros, coeff in ac_rle:
        coefficients.extend([0] * zeros)
        if coeff != 0:
            coefficients.append(coeff)
    # Ensure the coefficients list is 63 elements long
    if len(coefficients) < 63:
        coefficients.extend([0] * (63 - len(coefficients)))
    else:
        coefficients = coefficients[:63]
    return coefficients

# ====================== Image Padding and Blocking ======================

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

# ====================== Process Blocks with Differential DC Encoding ======================

def process_block(block, quant_table, prev_dc, log_file=None):
    dct_block = dct_2d(block)
    quantized_block = quantize(dct_block, quant_table)
    zigzagged_block = zigzag_order(quantized_block)
    dc_coefficient = zigzagged_block[0]
    dc_diff = dc_coefficient - prev_dc
    ac_coefficients = zigzagged_block[1:]
    rle_ac = run_length_encode(ac_coefficients)
    # Combine DC difference and AC coefficients
    symbols = [('DC', dc_diff)] + rle_ac + [('EOB',)]
    # Log the run-length encoded signal
    if log_file:
        log_file.write("Run-length encoded signal for block:\n")
        log_file.write(f"{symbols}\n")
    encoded_block, codes = huffman_encode(symbols, log_file)
    # Log the Huffman encoded signal
    if log_file:
        log_file.write("Huffman encoded signal for block:\n")
        log_file.write(f"{encoded_block}\n")
    return encoded_block, codes, dc_coefficient

def inverse_process_block(encoded_block, codes, quant_table, prev_dc, log_file=None):
    try:
        decoded_symbols = huffman_decode(encoded_block, codes, log_file)
        # Log the Huffman decoded symbols
        if log_file:
            log_file.write("Huffman decoded symbols for block:\n")
            log_file.write(f"{decoded_symbols}\n")
        # Separate DC difference and AC coefficients
        dc_diff = None
        ac_rle = []
        idx = 0
        while idx < len(decoded_symbols):
            symbol = decoded_symbols[idx]
            idx += 1
            if symbol[0] == 'DC':
                dc_diff = symbol[1]
            elif symbol[0] == 'AC':
                zeros, coeff = symbol[1], symbol[2]
                if (zeros, coeff) == (0, 0):
                    break  # End of Block
                ac_rle.append((zeros, coeff))
            elif symbol == ('EOB',):
                break
            else:
                raise ValueError(f"Unknown symbol {symbol}")
        if dc_diff is None:
            raise ValueError("DC coefficient difference missing")
        # Log the run-length decoded signal
        if log_file:
            log_file.write("Run-length decoded AC coefficients for block:\n")
            log_file.write(f"{ac_rle}\n")
        dc_coefficient = dc_diff + prev_dc
        ac_coefficients = run_length_decode(ac_rle)
        coefficients = [dc_coefficient] + ac_coefficients
        block = inverse_zigzag_order(coefficients)
        dequantized_block = dequantize(block, quant_table)
        idct_block = idct_2d(dequantized_block)
        return idct_block, dc_coefficient
    except Exception as e:
        if log_file:
            log_file.write(f"Error in inverse_process_block: {str(e)}\n")
        return np.zeros((8,8)), prev_dc  # Return a default block and previous DC

# ====================== JPEG Compression and Decompression ======================

def jpeg_compress(image_path, log_filename='compression_logs.txt'):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    ycbcr = rgb_to_ycbcr(image)
    padded = pad_image(ycbcr)
    height, width, _ = ycbcr.shape

    Y = padded[:,:,0]
    Cb = padded[:,:,1]
    Cr = padded[:,:,2]

    Y_blocks = block_split(Y)
    Cb_blocks = block_split(Cb)
    Cr_blocks = block_split(Cr)

    # Open log file for writing
    with open(log_filename, 'w') as log_file:
        # Process Y channel
        Y_compressed = []
        prev_dc = 0
        for idx, block in enumerate(Y_blocks):
            log_file.write(f"\nProcessing Y block {idx+1}/{len(Y_blocks)}:\n")
            encoded_block, codes, prev_dc = process_block(block, STANDARD_LUMINANCE_Q, prev_dc, log_file)
            Y_compressed.append((encoded_block, codes))

        # Process Cb channel
        Cb_compressed = []
        prev_dc = 0
        for idx, block in enumerate(Cb_blocks):
            log_file.write(f"\nProcessing Cb block {idx+1}/{len(Cb_blocks)}:\n")
            encoded_block, codes, prev_dc = process_block(block, STANDARD_CHROMINANCE_Q, prev_dc, log_file)
            Cb_compressed.append((encoded_block, codes))

        # Process Cr channel
        Cr_compressed = []
        prev_dc = 0
        for idx, block in enumerate(Cr_blocks):
            log_file.write(f"\nProcessing Cr block {idx+1}/{len(Cr_blocks)}:\n")
            encoded_block, codes, prev_dc = process_block(block, STANDARD_CHROMINANCE_Q, prev_dc, log_file)
            Cr_compressed.append((encoded_block, codes))
    compressed_data = {
        'Y': Y_compressed,
        'Cb': Cb_compressed,
        'Cr': Cr_compressed,
        'height': height,
        'width': width,
        'log_filename': log_filename  # Include log filename for decompression
    }
    compression_ratio = calculate_compression_ratio(image_path, compressed_data)
    print(f"Compression Ratio: {compression_ratio:.2f}")

    return {
        'Y': Y_compressed,
        'Cb': Cb_compressed,
        'Cr': Cr_compressed,
        'height': height,
        'width': width,
        'log_filename': log_filename  # Include log filename for decompression
    }
   
def jpeg_decompress(compressed):
    try:
        Y_compressed = compressed['Y']
        Cb_compressed = compressed['Cb']
        Cr_compressed = compressed['Cr']
        height, width = compressed['height'], compressed['width']
        log_filename = compressed.get('log_filename', 'compression_logs.txt')

        # Open log file for appending
        with open(log_filename, 'a') as log_file:
            # Reconstruct Y channel
            Y_reconstructed_blocks = []
            prev_dc = 0
            for idx, (encoded_block, codes) in enumerate(Y_compressed):
                log_file.write(f"\nDecompressing Y block {idx+1}/{len(Y_compressed)}:\n")
                idct_block, prev_dc = inverse_process_block(encoded_block, codes, STANDARD_LUMINANCE_Q, prev_dc, log_file)
                Y_reconstructed_blocks.append(idct_block)

            # Reconstruct Cb channel
            Cb_reconstructed_blocks = []
            prev_dc = 0
            for idx, (encoded_block, codes) in enumerate(Cb_compressed):
                log_file.write(f"\nDecompressing Cb block {idx+1}/{len(Cb_compressed)}:\n")
                idct_block, prev_dc = inverse_process_block(encoded_block, codes, STANDARD_CHROMINANCE_Q, prev_dc, log_file)
                Cb_reconstructed_blocks.append(idct_block)

            # Reconstruct Cr channel
            Cr_reconstructed_blocks = []
            prev_dc = 0
            for idx, (encoded_block, codes) in enumerate(Cr_compressed):
                log_file.write(f"\nDecompressing Cr block {idx+1}/{len(Cr_compressed)}:\n")
                idct_block, prev_dc = inverse_process_block(encoded_block, codes, STANDARD_CHROMINANCE_Q, prev_dc, log_file)
                Cr_reconstructed_blocks.append(idct_block)

        # Merge the reconstructed blocks into channels
        padded_height = (height + 7) // 8 * 8
        padded_width = (width + 7) // 8 * 8

        Y_channel = block_merge(Y_reconstructed_blocks, padded_height, padded_width)
        Cb_channel = block_merge(Cb_reconstructed_blocks, padded_height, padded_width)
        Cr_channel = block_merge(Cr_reconstructed_blocks, padded_height, padded_width)

        # Crop to the original dimensions
        Y_channel = Y_channel[:height, :width]
        Cb_channel = Cb_channel[:height, :width]
        Cr_channel = Cr_channel[:height, :width]

        ycbcr = np.stack((Y_channel, Cb_channel, Cr_channel), axis=2)
        rgb = ycbcr_to_rgb(ycbcr)
        return rgb
    except Exception as e:
        print(f"Error in jpeg_decompress: {str(e)}")
        return None

# ====================== Main ======================

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description='JPEG Compression Pipeline')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
    args = parser.parse_args()

    try:
        start_time = time.time()
        compressed = jpeg_compress(args.input_image)
        compressed_time = time.time()
        print(f"Compression completed in {compressed_time - start_time:.2f} seconds.")

        reconstructed = jpeg_decompress(compressed)

        if reconstructed is not None:
            decompressed_time = time.time()
            print(f"Decompression completed in {decompressed_time - compressed_time:.2f} seconds.")
            cv2.imwrite(args.output_image, reconstructed)
            print(f"\nReconstructed image saved to {args.output_image}")
            print("Logs have been saved to compression_logs.txt")
        else:
            print("Failed to reconstruct image.")
    except Exception as e:
        print(f"Error: {str(e)}")
