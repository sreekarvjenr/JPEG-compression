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
    Decode run-length encoded data into a flat list of exactly 64 coefficients.
    """
    # Start with the DC coefficient
    coefficients = [dc]
    
    for item in rle:
        # Check if the item is a tuple (run-length encoding of zeros and a non-zero coefficient)
        if isinstance(item, tuple):
            zeros, coeff = item
            # Append the specified number of zeros
            coefficients.extend([0] * zeros)
            # Append the coefficient only if it's non-zero (to ensure accurate block size)
            if coeff != 0:
                coefficients.append(coeff)
        else:
            # If the item is already a scalar, append it directly
            coefficients.append(item)
    
    # Pad or trim the coefficients list to ensure it has exactly 64 elements
    coefficients = coefficients[:64] + [0] * (64 - len(coefficients))

    # Validation check: Ensure there are no tuples remaining in the output list
    # for idx, value in enumerate(coefficients):
    #     if isinstance(value, tuple):
    #         print(f"Error: Non-scalar value found at index {idx}: {value}")
    #         raise ValueError(f"Non-scalar value found at index {idx} in coefficients: {value}")

    return coefficients




