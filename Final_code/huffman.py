import heapq
from collections import defaultdict
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