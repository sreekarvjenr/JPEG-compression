# JPEG Compression and Decompression Pipeline

This project implements a basic JPEG compression and decompression pipeline using Python. The implementation covers essential aspects of the JPEG algorithm, including Discrete Cosine Transform (DCT), quantization, zigzag ordering, run-length encoding (RLE), Huffman encoding, and decoding for compressing and reconstructing images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Compression](#compression)
  - [Decompression](#decompression)
- [Components](#components)
  - [Color Space Conversion](#color-space-conversion)
  - [DCT and IDCT](#dct-and-idct)
  - [Quantization](#quantization)
  - [Zigzag Ordering](#zigzag-ordering)
  - [Run-Length Encoding (RLE)](#run-length-encoding-rle)
  - [Huffman Encoding](#huffman-encoding)
- [Example Commands](#example-commands)
- [Acknowledgements](#acknowledgements)

## Overview

JPEG (Joint Photographic Experts Group) is a widely-used image compression standard. This project implements the main steps of JPEG compression and decompression in a straightforward manner, providing insights into how image data is efficiently transformed, quantized, and encoded to reduce file sizes.

## Features

- Conversion of RGB images to YCbCr color space for luminance-chrominance separation.
- Block-based Discrete Cosine Transform (DCT) and Inverse DCT (IDCT).
- Standard JPEG quantization tables for luminance and chrominance channels.
- Efficient zigzag ordering, run-length encoding (RLE), and Huffman encoding for data compression.
- Real-time encoding and decoding of image data.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have OpenCV (`cv2`) and NumPy installed.

## Usage

### Compression

To compress an image:
```bash
python main.py <input_image> <compressed_data.bin> <output_image>
```
This will compress the image and save the reconstructed decompressed image.

## Components

### Color Space Conversion

- Converts RGB images to YCbCr color space to separate luminance (Y) from chrominance (Cb, Cr) channels.
- This separation leverages human perception, focusing compression on less sensitive chrominance channels.

### DCT and IDCT

- Applies an 8x8 block-based Discrete Cosine Transform (DCT) to convert spatial image data into frequency data.
- High-frequency coefficients are compressed more since they contribute less to visual perception.
- Inverse DCT (IDCT) reconstructs spatial data from the compressed frequency data.

### Quantization

- Reduces precision of DCT coefficients using standard quantization tables for luminance and chrominance.
- Improves compression by emphasizing visually significant data.

### Zigzag Ordering

- Reorders coefficients from an 8x8 block into a one-dimensional list using a zigzag pattern for better run-length encoding.

### Run-Length Encoding (RLE)

- Encodes sequences of zero coefficients efficiently by capturing the number of consecutive zeros before a non-zero value.
- Separates DC and AC components of image blocks for efficient encoding.

### Huffman Encoding

- Encodes data using variable-length codes, with shorter codes assigned to more frequent symbols.
- Improves compression efficiency by minimizing the number of bits used.

## Example Commands

### Compress an Image
```bash
python main.py input_image.jpg compressed_data.bin output_image.jpg
```

## Acknowledgements

This project is inspired by the core concepts of JPEG compression, demonstrating the essential techniques for compressing and reconstructing image data in an efficient manner.