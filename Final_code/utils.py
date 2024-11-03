import numpy as np
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
