"""Part 1.1: Convolutions from Scratch!

Implements 2D convolution with (a) four nested for loops and (b) two nested
for loops (kernel taps only, output plane vectorized), checks both against
scipy.signal.convolve2d, times all three, and applies a 9x9 box filter and
the finite difference operators D_x, D_y to a self-portrait.
"""

import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d

from conv import conv2d_2loop, conv2d_4loop

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def to_uint8(im: np.ndarray) -> np.ndarray:
    """Min-max normalize a float array to [0, 255] uint8 for saving/display."""
    lo, hi = im.min(), im.max()
    if hi - lo < 1e-12:
        return np.zeros_like(im, dtype=np.uint8)
    return (255 * (im - lo) / (hi - lo)).astype(np.uint8)


def main():
    img = (
        cv2.imread(str(DATA / "self_portrait.jpg"), cv2.IMREAD_GRAYSCALE).astype(
            np.float64
        )
        / 255.0
    )
    cv2.imwrite(str(OUT / "1_1_self_portrait_gray.jpg"), to_uint8(img))

    box9 = np.ones((9, 9)) / 81.0
    Dx = np.array([[1.0, -1.0]])
    Dy = np.array([[1.0], [-1.0]])

    # Correctness + timing comparison on a small crop (the naive 4-loop
    # version is far too slow in pure Python to run on the full image)
    small = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)

    results = {}
    timings = {}

    t0 = time.perf_counter()
    results["4loop"] = conv2d_4loop(small, box9)
    timings["4loop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    results["2loop"] = conv2d_2loop(small, box9)
    timings["2loop"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    results["scipy"] = convolve2d(
        small, box9, mode="same", boundary="fill", fillvalue=0
    )
    timings["scipy"] = time.perf_counter() - t0

    report = [f"Test image: {small.shape}, kernel: {box9.shape} (mode='same')"]
    report += [f"{name}: {t * 1000:.3f} ms" for name, t in timings.items()]
    report.append(
        f"max|4loop - scipy| = {np.abs(results['4loop'] - results['scipy']).max():.3e}"
    )
    report.append(
        f"max|2loop - scipy| = {np.abs(results['2loop'] - results['scipy']).max():.3e}"
    )

    print("\n".join(report))
    (OUT / "1_1_timing.txt").write_text("\n".join(report) + "\n")

    # --- Apply to the full self-portrait (use the vectorized 2-loop version
    # for speed; it is numerically identical to the 4-loop version) ---
    boxed = conv2d_2loop(img, box9)
    dx = conv2d_2loop(img, Dx)
    dy = conv2d_2loop(img, Dy)

    cv2.imwrite(str(OUT / "1_1_box_filtered.jpg"), to_uint8(boxed))
    cv2.imwrite(str(OUT / "1_1_dx.jpg"), to_uint8(dx))
    cv2.imwrite(str(OUT / "1_1_dy.jpg"), to_uint8(dy))

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
