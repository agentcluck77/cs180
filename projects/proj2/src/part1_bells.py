"""Part 1 Bells & Whistles: gradient orientation visualized in HSV.

Hue encodes orientation (a genuinely cyclic quantity, so it wraps around
exactly like hue does), and brightness is scaled by gradient magnitude so
flat regions (no well-defined orientation) fade to black instead of
showing meaningless color.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def main():
    img = cv2.imread(str(DATA / "cameraman.png"), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    Dx = np.array([[1.0, -1.0]])
    Dy = np.array([[1.0], [-1.0]])
    dx = convolve2d(img, Dx, mode="same", boundary="fill", fillvalue=0)
    dy = convolve2d(img, Dy, mode="same", boundary="fill", fillvalue=0)

    # orientation via the sign/ratio of dx, dy directly (no built-in angle
    # function): standard atan2 identity, computed from first principles.
    angle = np.where(
        dx >= 0,
        np.arcsin(dy / (np.sqrt(dx**2 + dy**2) + 1e-12)),
        np.pi - np.arcsin(dy / (np.sqrt(dx**2 + dy**2) + 1e-12)),
    )
    angle_norm = (angle + np.pi) / (2 * np.pi) % 1.0  # -> [0, 1), cyclic

    mag = np.sqrt(dx**2 + dy**2)
    mag_norm = mag / (mag.max() + 1e-12)

    cmap = plt.get_cmap("hsv")  # cyclic colormap, matches orientation's wraparound
    rgb = cmap(angle_norm)[..., :3]
    rgb = rgb * mag_norm[..., None]  # fade flat regions to black

    plt.imsave(OUT / "1_bells_gradient_orientation.png", rgb)

    # legend: a ring showing which color maps to which orientation
    yy, xx = np.mgrid[-1:1:200j, -1:1:200j]
    ring_angle = np.where(xx >= 0, np.arcsin(yy / (np.sqrt(xx**2 + yy**2) + 1e-12)),
                          np.pi - np.arcsin(yy / (np.sqrt(xx**2 + yy**2) + 1e-12)))
    ring_norm = (ring_angle + np.pi) / (2 * np.pi) % 1.0
    ring_mask = (xx**2 + yy**2) <= 1.0
    ring_rgb = cmap(ring_norm)[..., :3]
    ring_rgb[~ring_mask] = 1.0
    plt.imsave(OUT / "1_bells_orientation_legend.png", ring_rgb)

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
