"""Part 1.2: Finite Difference Operator.

Partial derivatives of the cameraman image via D_x, D_y, the gradient
magnitude image, and a binarized edge image.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def to_uint8(im: np.ndarray) -> np.ndarray:
    lo, hi = im.min(), im.max()
    if hi - lo < 1e-12:
        return np.zeros_like(im, dtype=np.uint8)
    return (255 * (im - lo) / (hi - lo)).astype(np.uint8)


def main():
    img = cv2.imread(str(DATA / "cameraman.png"), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    cv2.imwrite(str(OUT / "1_2_cameraman_gray.jpg"), to_uint8(img))

    Dx = np.array([[1.0, -1.0]])
    Dy = np.array([[1.0], [-1.0]])

    dx = convolve2d(img, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(img, Dy, mode="same", boundary="fill", fillvalue=0)
    grad_mag = np.sqrt(dx**2 + dy**2)

    cv2.imwrite(str(OUT / "1_2_dx.jpg"), to_uint8(dx))
    cv2.imwrite(str(OUT / "1_2_dy.jpg"), to_uint8(dy))
    cv2.imwrite(str(OUT / "1_2_grad_mag.jpg"), to_uint8(grad_mag))

    print("gradient magnitude stats: min=%.4f max=%.4f mean=%.4f" % (grad_mag.min(), grad_mag.max(), grad_mag.mean()))

    # Try a few thresholds to pick a good one qualitatively.
    candidates = [0.10, 0.15, 0.20, 0.25, 0.30]
    for t in candidates:
        edges = (grad_mag > t).astype(np.uint8) * 255
        cv2.imwrite(str(OUT / f"1_2_edges_t{t:.2f}.jpg"), edges)

    # Chosen threshold (see writeup): suppresses sensor/texture noise on the
    # grass and coat while keeping the camera, tripod, and skyline edges.
    threshold = 0.25
    edges = (grad_mag > threshold).astype(np.uint8) * 255
    cv2.imwrite(str(OUT / "1_2_edges_final.jpg"), edges)

    print(f"Chosen threshold: {threshold}")
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
