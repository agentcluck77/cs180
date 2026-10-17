"""2D convolution from scratch"""

import numpy as np


def conv2d_4loop(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution (flips the kernel), four nested for loops. Same-size output."""
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)))
    flipped = kernel[::-1, ::-1]
    out = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            acc = 0.0
            for m in range(kh):
                for n in range(kw):
                    acc += padded[i + m, j + n] * flipped[m, n]
            out[i, j] = acc
    return out


def conv2d_2loop(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same convolution, but the two loops over output pixels are vectorized
    away with slicing, leaving only the two loops over kernel taps."""
    H, W = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)))
    flipped = kernel[::-1, ::-1]
    out = np.zeros((H, W), dtype=np.float64)
    for m in range(kh):
        for n in range(kw):
            out += flipped[m, n] * padded[m : m + H, n : n + W]
    return out
