# huffman.py
import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    # Define comparison operators for the priority queue
    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_table(data):
    """
    Build a frequency table for the symbols in data.
    """
    freq = defaultdict(int)
    for symbol in data:
        freq[symbol] += 1
    return freq

def build_huffman_tree(freq_table):
    """
    Build the Huffman tree and return the root.
    """
    heap = []
    for symbol, freq in freq_table.items():
        node = HuffmanNode(symbol, freq)
        heapq.heappush(heap, node)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    
    return heap[0] if heap else None

def build_codes(node, prefix="", codebook={}):
    """
    Build the Huffman codes for each symbol.
    """
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data):
    """
    Encode data using Huffman Coding.
    """
    freq_table = build_frequency_table(data)
    root = build_huffman_tree(freq_table)
    codes = build_codes(root)
    encoded_data = ''.join([codes[symbol] for symbol in data])
    return encoded_data, codes

def huffman_decode(encoded_data, codes, num_blocks=1024*4):
    """
    Decode Huffman encoded data for multiple blocks.
    """
    # Reverse the codes dictionary to map encoded strings back to symbols
    reversed_codes = {v: k for k, v in codes.items()}
    
    decoded_blocks = []
    current_code = ""
    decoded_data = []
    block_count = 0
    
    # Decode each bit in the encoded data
    for bit in encoded_data:
        current_code += bit
        # If current_code matches a code, decode it
        if current_code in reversed_codes:
            symbol = reversed_codes[current_code]
            # Check for the (0, 0) EOB marker only if symbol is a tuple
            if isinstance(symbol, tuple) and symbol == (0, 0):
                # Calculate and add trailing zeros to reach 64 elements
                zeros_needed = 64 - len(decoded_data)
                decoded_data.extend([0] * zeros_needed)
                decoded_blocks.append(decoded_data)
                 # Add completed block
                decoded_data = []  # Reset for the next block
                block_count += 1
                if block_count == num_blocks:
                    break  # Stop if we reach the desired number of blocks
            else:
                 
                # Add the decoded symbol to the current block
                decoded_data.append(symbol)
            
            # Reset current_code for the next sequence
            current_code = ""
    
   

    return decoded_blocks



