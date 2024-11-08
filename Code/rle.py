import numpy as np
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
