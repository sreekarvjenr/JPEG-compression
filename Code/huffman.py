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

def huffman_decode(encoded_data, codes):
    """
    Decode Huffman encoded data.
    """
    reversed_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_data = []
    for bit in encoded_data:
        current_code += bit
        if current_code in reversed_codes:
            decoded_data.append(reversed_codes[current_code])
            current_code = ""
    return decoded_data
