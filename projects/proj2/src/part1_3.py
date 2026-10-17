"""Part 1.3: Derivative of Gaussian (DoG) Filter."""

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


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    k1d = cv2.getGaussianKernel(ksize, sigma)  # (ksize, 1)
    return k1d @ k1d.T  # (ksize, ksize)


def main():
    img = cv2.imread(str(DATA / "cameraman.png"), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    Dx = np.array([[1.0, -1.0]])
    Dy = np.array([[1.0], [-1.0]])

    ksize, sigma = 9, 2.0
    G = gaussian_kernel_2d(ksize, sigma)
    cv2.imwrite(str(OUT / "1_3_gaussian_kernel.jpg"), to_uint8(cv2.resize(G, (200, 200), interpolation=cv2.INTER_NEAREST)))

    # --- Step 1: blur then finite-difference ---
    blurred = convolve2d(img, G, mode="same", boundary="fill", fillvalue=0)
    cv2.imwrite(str(OUT / "1_3_blurred.jpg"), to_uint8(blurred))

    dx_blurred = convolve2d(blurred, Dx, mode="same", boundary="fill", fillvalue=0)
    dy_blurred = convolve2d(blurred, Dy, mode="same", boundary="fill", fillvalue=0)
    grad_mag_blurred = np.sqrt(dx_blurred**2 + dy_blurred**2)
    cv2.imwrite(str(OUT / "1_3_dx_blurred.jpg"), to_uint8(dx_blurred))
    cv2.imwrite(str(OUT / "1_3_dy_blurred.jpg"), to_uint8(dy_blurred))
    cv2.imwrite(str(OUT / "1_3_grad_mag_blurred.jpg"), to_uint8(grad_mag_blurred))

    print("blurred grad mag stats: min=%.4f max=%.4f mean=%.4f" % (
        grad_mag_blurred.min(), grad_mag_blurred.max(), grad_mag_blurred.mean()))

    candidates = [0.05, 0.08, 0.10, 0.12, 0.15]
    for t in candidates:
        edges = (grad_mag_blurred > t).astype(np.uint8) * 255
        cv2.imwrite(str(OUT / f"1_3_edges_t{t:.2f}.jpg"), edges)

    threshold_blurred = 0.10
    edges_blurred = (grad_mag_blurred > threshold_blurred).astype(np.uint8) * 255
    cv2.imwrite(str(OUT / "1_3_edges_final.jpg"), edges_blurred)
    print(f"Chosen threshold (blurred): {threshold_blurred}")

    # --- Step 2: single-convolution Derivative-of-Gaussian filters ---
    DoG_x = convolve2d(G, Dx, mode="full")
    DoG_y = convolve2d(G, Dy, mode="full")
    cv2.imwrite(str(OUT / "1_3_dog_x_kernel.jpg"), to_uint8(cv2.resize(DoG_x, (220, 200), interpolation=cv2.INTER_NEAREST)))
    cv2.imwrite(str(OUT / "1_3_dog_y_kernel.jpg"), to_uint8(cv2.resize(DoG_y, (200, 220), interpolation=cv2.INTER_NEAREST)))

    dx_dog = convolve2d(img, DoG_x, mode="same", boundary="fill", fillvalue=0)
    dy_dog = convolve2d(img, DoG_y, mode="same", boundary="fill", fillvalue=0)
    grad_mag_dog = np.sqrt(dx_dog**2 + dy_dog**2)
    cv2.imwrite(str(OUT / "1_3_dx_dog.jpg"), to_uint8(dx_dog))
    cv2.imwrite(str(OUT / "1_3_dy_dog.jpg"), to_uint8(dy_dog))
    cv2.imwrite(str(OUT / "1_3_grad_mag_dog.jpg"), to_uint8(grad_mag_dog))

    # Verify single-step DoG convolution matches the two-step blur-then-diff
    # result (ignore a border strip where the two padding conventions differ).
    b = ksize  # border to ignore
    diff_x = np.abs(dx_dog[b:-b, b:-b] - dx_blurred[b:-b, b:-b]).max()
    diff_y = np.abs(dy_dog[b:-b, b:-b] - dy_blurred[b:-b, b:-b]).max()
    print(f"max|DoG_x conv - blur-then-Dx| (interior) = {diff_x:.3e}")
    print(f"max|DoG_y conv - blur-then-Dy| (interior) = {diff_y:.3e}")

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
