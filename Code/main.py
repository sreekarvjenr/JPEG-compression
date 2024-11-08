import cv2
import numpy as np
from color_space import rgb_to_ycbcr, ycbcr_to_rgb
from utils import pad_image, block_split, block_merge
from dct import dct_2d, idct_2d
from quantization import quantize, dequantize, STANDARD_LUMINANCE_Q, STANDARD_CHROMINANCE_Q
from rle import run_length_encode, run_length_decode
from huffman import huffman_encode, huffman_decode
import os
import pickle
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

def save_compressed_data_to_bin(compressed_data, bin_file_path):
    """Saves the compressed data to a .bin file."""
    with open(bin_file_path, 'wb') as bin_file:
        pickle.dump(compressed_data, bin_file)
    print(f"Compressed data saved to {bin_file_path}")

def jpeg_compress(image_path, bin_file_path, log_filename='compression_logs.txt'):
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
    
    # Save the compressed data to a .bin file
    save_compressed_data_to_bin(compressed_data, bin_file_path)
    
    compression_ratio = calculate_compression_ratio(image_path, compressed_data)
    print(f"Compression Ratio: {compression_ratio:.2f}")
    
    return compressed_data
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
    parser.add_argument('output_bin', type=str, help='Path to save the compressed .bin file')
    parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
    args = parser.parse_args()

    try:
        start_time = time.time()
        compressed = jpeg_compress(args.input_image, args.output_bin)
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