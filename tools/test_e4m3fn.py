#!/usr/bin/env python3
"""Small dependency-free check for the cache builders' FP8 encoder."""

import numpy as np

from quantize_dflash2 import encode_e4m3fn


values = np.array([
    0.0, -0.0, 2.0 ** -10, 3.0 * 2.0 ** -10,
    2.0 ** -9, -(2.0 ** -9), 2.0 ** -6,
    1.0, 1.0625, 1.1875, 448.0, -448.0,
], dtype=np.float32)
expected = np.array([
    0x00, 0x80, 0x00, 0x02, 0x01, 0x81, 0x08,
    0x38, 0x38, 0x3A, 0x7E, 0xFE,
], dtype=np.uint8)
actual = encode_e4m3fn(values)
assert np.array_equal(actual, expected), (values, expected, actual)
print("E4M3FN encoder: ok")
