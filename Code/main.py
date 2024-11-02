# main.py
import cv2
import numpy as np
from color_space import rgb_to_ycbcr, ycbcr_to_rgb
from utils import pad_image, block_split, block_merge
from dct import dct_2d, idct_2d
from quantization import quantize, dequantize, STANDARD_LUMINANCE_Q, STANDARD_CHROMINANCE_Q
from rle import run_length_encode, run_length_decode
from huffman import huffman_encode, huffman_decode

def zigzag_order(block):
    """
    Rearrange the 8x8 block into a zig-zag order list.
    """
    zigzag_indices = [
        (0,0),
        (0,1),(1,0),
        (2,0),(1,1),(0,2),
        (0,3),(1,2),(2,1),(3,0),
        (4,0),(3,1),(2,2),(1,3),(0,4),
        (0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
        (6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
        (0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0),
        (7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),
        (2,7),(3,6),(4,5),(5,4),(6,3),(7,2),
        (7,3),(6,4),(5,5),(4,6),(3,7),
        (4,7),(5,6),(6,5),(7,4),
        (7,5),(6,6),(5,7),
        (6,7),(7,6),
        (7,7)
    ]
    return [block[i,j] for i,j in zigzag_indices]

# def inverse_zigzag_order(coefficients):
#     """
#     Rearrange the zig-zag ordered list back into an 8x8 block.
#     """
#     block = np.zeros((8,8), dtype=np.float32)
#     zigzag_indices = [
#         (0,0),
#         (0,1),(1,0),
#         (2,0),(1,1),(0,2),
#         (0,3),(1,2),(2,1),(3,0),
#         (4,0),(3,1),(2,2),(1,3),(0,4),
#         (0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
#         (6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
#         (0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0),
#         (7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),
#         (2,7),(3,6),(4,5),(5,4),(6,3),(7,2),
#         (7,3),(6,4),(5,5),(4,6),(3,7),
#         (4,7),(5,6),(6,5),(7,4),
#         (7,5),(6,6),(5,7),
#         (6,7),(7,6),
#         (7,7)
#     ]
#     print(coefficients)
#     for idx, (i,j) in enumerate(zigzag_indices):
#         if idx < len(coefficients):
#             block[i,j] = coefficients[idx]
#     return block
import numpy as np

def inverse_zigzag_order(coefficients):
    """
    Rearrange the zig-zag ordered list back into an 8x8 block.
    """
    block = np.zeros((8, 8), dtype=np.float32)
    zigzag_indices = [
        (0,0),
        (0,1),(1,0),
        (2,0),(1,1),(0,2),
        (0,3),(1,2),(2,1),(3,0),
        (4,0),(3,1),(2,2),(1,3),(0,4),
        (0,5),(1,4),(2,3),(3,2),(4,1),(5,0),
        (6,0),(5,1),(4,2),(3,3),(2,4),(1,5),(0,6),
        (0,7),(1,6),(2,5),(3,4),(4,3),(5,2),(6,1),(7,0),
        (7,1),(6,2),(5,3),(4,4),(3,5),(2,6),(1,7),
        (2,7),(3,6),(4,5),(5,4),(6,3),(7,2),
        (7,3),(6,4),(5,5),(4,6),(3,7),
        (4,7),(5,6),(6,5),(7,4),
        (7,5),(6,6),(5,7),
        (6,7),(7,6),
        (7,7)
    ]
    #print(coefficients)
    for idx, (i, j) in enumerate(zigzag_indices):
        if idx < len(coefficients):
            value = coefficients[idx]
            if isinstance(value, tuple):
                block[i,j]=value[1] 
                continue
            # Allow Python ints, floats, and all numpy scalar types
            if isinstance(value, (int, float, np.generic)):
                block[i, j] = value
            else:
                print(f"Error: Non-scalar value found at index {idx}: {value} (type: {type(value)})")
                raise ValueError(f"Expected scalar at index {idx}, but got {type(value)}: {value}")
    return block


def process_channel(channel, quant_table):
    """
    Process a single color channel: DCT, Quantization, Zigzag, RLE, Huffman Encoding.
    """
    blocks = block_split(channel)
    dct_blocks = [dct_2d(block) for block in blocks]
    quantized_blocks = [quantize(block, quant_table) for block in dct_blocks]
    zigzagged = [zigzag_order(block) for block in quantized_blocks]
    
    # Flatten all zigzagged blocks for Huffman encoding
    flat_coeffs = []
    for block in zigzagged:
        dc, rle = run_length_encode(block)
        flat_coeffs.append(dc)
        for item in rle:
            flat_coeffs.append(item)
    
    # Huffman Encoding
    encoded_data, codes = huffman_encode(flat_coeffs)
    
    return {
        'encoded_data': encoded_data,
        'codes': codes,
        'quant_table': quant_table,
        'blocks_count': len(blocks)
    }
def inverse_process_channel(data, height, width):
    """
    Inverse process a single color channel: Huffman Decoding, RLE, Inverse Zigzag, Dequantization, IDCT.
    """
    encoded_data = data['encoded_data']
    codes = data['codes']
    quant_table = data['quant_table']
    blocks_count = data['blocks_count']
    
    # Huffman Decoding for all blocks
    flat_blocks = huffman_decode(encoded_data, codes, num_blocks=blocks_count)
    
    # Reconstruct blocks
    blocks = []
    for flat_coeffs in flat_blocks:
        print()
        # Ensure each block is exactly 64 scalar values
        if len(flat_coeffs) != 64 or any(isinstance(x, list) for x in flat_coeffs):
            raise ValueError("Decoded block is not a flat list of 64 scalar values.")
        
        # Convert flat coefficients to 8x8 block
        block = inverse_zigzag_order(flat_coeffs)
        dequant = dequantize(block, quant_table)
        idct_block = idct_2d(dequant)
        blocks.append(idct_block)

    # Merge blocks into the channel
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    channel = block_merge(blocks, padded_height, padded_width)
    
    # Crop to original size
    channel = channel[:height, :width]
    return channel

# def inverse_process_channel(data, height, width):
#     """
#     Inverse process a single color channel: Huffman Decoding, RLE, Inverse Zigzag, Dequantization, IDCT.
#     """
#     encoded_data = data['encoded_data']
#     codes = data['codes']
#     quant_table = data['quant_table']
#     blocks_count = data['blocks_count']
    
#     # Huffman Decoding
#     flat_coeffs = huffman_decode(encoded_data, codes)
    
#     # Reconstruct blocks
#     blocks = []
#     idx = 0
#     print((blocks_count))
#     for _ in range(blocks_count):
#         if idx >= len(flat_coeffs):
#             break
#         # DC coefficient
#         dc = flat_coeffs[idx]
#         idx += 1
        
#         # RLE Decoding
#         rle = []
#         while idx < len(flat_coeffs):
#             symbol = flat_coeffs[idx]
#             idx += 1
#             # Stop if (0, 0) marker is encountered
#             if symbol == (0, 0):
#                 rle.append((0, 0))
#                 break
#             rle.append((0, symbol) if symbol != 0 else (0, 0))

#         # Decode RLE into coefficients
#         coeffs = run_length_decode(dc, rle)

#         # Perform Inverse Zigzag, Dequantization, and IDCT
#         block = inverse_zigzag_order(coeffs)
#         dequant = dequantize(block, quant_table)
#         idct_block = idct_2d(dequant)
#         blocks.append(idct_block)
        
#     # Merge blocks into the channel
#     padded_height = (height + 7) // 8 * 8
#     padded_width = (width + 7) // 8 * 8
#     channel = block_merge(blocks, padded_height, padded_width)
    
#     # Crop to original size
#     channel = channel[:height, :width]
#     return channel


def jpeg_compress(image_path):
    """
    Compress the image and return compressed data.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    ycbcr = rgb_to_ycbcr(image)
    padded = pad_image(ycbcr)
    height, width, _ = ycbcr.shape

    # Split channels
    Y = padded[:,:,0]
    Cb = padded[:,:,1]
    Cr = padded[:,:,2]

    # Process Y channel
    Y_data = process_channel(Y, STANDARD_LUMINANCE_Q)
    # Process Cb and Cr channels
    Cb_data = process_channel(Cb, STANDARD_CHROMINANCE_Q)
    Cr_data = process_channel(Cr, STANDARD_CHROMINANCE_Q)

    compressed = {
        'Y': Y_data,
        'Cb': Cb_data,
        'Cr': Cr_data,
        'height': height,
        'width': width
    }
    return compressed

def jpeg_decompress(compressed):
    """
    Decompress the data and return the reconstructed RGB image.
    """
    print("dimensions")
    print(compressed['height'])
    print(compressed['width'])
    print()
    Y = inverse_process_channel(compressed['Y'], compressed['height'], compressed['width'])
    Cb = inverse_process_channel(compressed['Cb'], compressed['height'], compressed['width'])
    Cr = inverse_process_channel(compressed['Cr'], compressed['height'], compressed['width'])
    
    # Merge channels
    ycbcr = np.stack((Y, Cb, Cr), axis=2)
    rgb = ycbcr_to_rgb(ycbcr)
    return rgb

def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='JPEG Compression Pipeline')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('output_image', type=str, help='Path to save the decompressed image')
    args = parser.parse_args()

    start_time = time.time()
    compressed = jpeg_compress(args.input_image)
    compressed_time = time.time()
    print(f"Compression completed in {compressed_time - start_time:.2f} seconds.")
    #print(compressed)
    reconstructed = jpeg_decompress(compressed)
    decompressed_time = time.time()
    print(f"Decompression completed in {decompressed_time - compressed_time:.2f} seconds.")

    cv2.imwrite(args.output_image, reconstructed)
    print(f"Reconstructed image saved to {args.output_image}")

if __name__ == "__main__":
    main()
