"""Part 2.3: Gaussian and Laplacian stacks (pyramids without downsampling)."""

import numpy as np

from utils import gaussian_blur


def gaussian_stack(img: np.ndarray, levels: int, sigma: float) -> list[np.ndarray]:
    stack = [img]
    for _ in range(1, levels):
        stack.append(gaussian_blur(stack[-1], sigma))
    return stack


def laplacian_stack(gstack: list[np.ndarray]) -> list[np.ndarray]:
    """Differences between consecutive Gaussian levels; last level is the
    final lowpass residual. Sums back to gstack[0] exactly (telescoping)."""
    lstack = [gstack[i] - gstack[i + 1] for i in range(len(gstack) - 1)]
    lstack.append(gstack[-1])
    return lstack
