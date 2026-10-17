"""Small helpers shared across Part 2 scripts."""

from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def to_uint8(im: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8 for saving/display."""
    lo, hi = im.min(), im.max()
    if hi - lo < 1e-12:
        return np.zeros_like(im, dtype=np.uint8)
    return (255 * (im - lo) / (hi - lo)).astype(np.uint8)


def clip_uint8(im: np.ndarray) -> np.ndarray:
    """Clip a [0, 1]-scaled image to that range and convert to uint8."""
    return (np.clip(im, 0.0, 1.0) * 255).astype(np.uint8)


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    k1d = cv2.getGaussianKernel(ksize, sigma)
    return k1d @ k1d.T


def load_color(path) -> np.ndarray:
    """Returns an RGB float image in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def save_color(path, im_rgb_float: np.ndarray) -> None:
    bgr = cv2.cvtColor(clip_uint8(im_rgb_float), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Fast separable Gaussian blur (same result as full 2D conv with the
    outer-product kernel, since a Gaussian is separable): g(x,y) = g(x) g(y)."""
    ksize = int(2 * round(3 * sigma) + 1)
    k1d = cv2.getGaussianKernel(ksize, sigma).flatten()
    if img.ndim == 2:
        return cv2.sepFilter2D(img, -1, k1d, k1d)
    return np.stack(
        [cv2.sepFilter2D(img[..., c], -1, k1d, k1d) for c in range(img.shape[-1])], axis=-1
    )


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb @ np.array([0.2989, 0.5870, 0.1140])


def conv2d_per_channel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size convolution of each channel of an (H, W, C) image with a 2D kernel."""
    if img.ndim == 2:
        return convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)
    channels = [convolve2d(img[..., c], kernel, mode="same", boundary="fill", fillvalue=0)
                for c in range(img.shape[-1])]
    return np.stack(channels, axis=-1)
