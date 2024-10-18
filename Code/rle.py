# rle.py
def run_length_encode(block):
    """
    Perform run-length encoding on a zig-zag ordered list of DCT coefficients.
    """
    rle = []
    zero_count = 0
    # Skip the DC coefficient for now
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
    return (dc, rle)

def run_length_decode(dc, rle):
    """
    Decode run-length encoded data.
    """
    ac = []
    for (zeros, coeff) in rle:
        if (zeros, coeff) == (0, 0):
            ac.extend([0] * (64 -1 - len(ac)))  # Fill the rest with zeros
            break
        ac.extend([0] * zeros)
        ac.append(coeff)
    # Ensure the list has exactly 63 AC coefficients
    ac = ac[:63] + [0]*(63 - len(ac))
    return [dc] + ac
