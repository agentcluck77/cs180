"""Part 2.3: Gaussian and Laplacian Stacks, applied to the apple/orange pair.

Recreates the spirit of Szeliski Figure 3.42: a grid of Gaussian and
Laplacian stack levels for both images.
"""

import matplotlib.pyplot as plt
import numpy as np

from stacks import gaussian_stack, laplacian_stack
from utils import DATA, OUT, load_color

LEVELS = 5
SIGMA = 4
SHOW_LEVELS = [0, 2, 4]


def norm01(im: np.ndarray) -> np.ndarray:
    lo, hi = im.min(), im.max()
    return (im - lo) / (hi - lo + 1e-12)


def main():
    apple = load_color(DATA / "apple_orange/apple.jpeg")
    orange = load_color(DATA / "apple_orange/orange.jpeg")

    g_apple = gaussian_stack(apple, LEVELS, SIGMA)
    g_orange = gaussian_stack(orange, LEVELS, SIGMA)
    l_apple = laplacian_stack(g_apple)
    l_orange = laplacian_stack(g_orange)

    # sanity check: stacks telescope back to the original image
    recon_apple = sum(l_apple)
    print("max|sum(laplacian_apple) - apple| =", np.abs(recon_apple - apple).max())

    fig, axes = plt.subplots(len(SHOW_LEVELS), 4, figsize=(12, 3 * len(SHOW_LEVELS)))
    col_titles = ["Gaussian: apple", "Gaussian: orange", "Laplacian: apple", "Laplacian: orange"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)
    for r, lvl in enumerate(SHOW_LEVELS):
        axes[r, 0].imshow(np.clip(g_apple[lvl], 0, 1))
        axes[r, 1].imshow(np.clip(g_orange[lvl], 0, 1))
        axes[r, 2].imshow(norm01(l_apple[lvl]))
        axes[r, 3].imshow(norm01(l_orange[lvl]))
        axes[r, 0].set_ylabel(f"level {lvl}")
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "2_3_stacks_grid.png", dpi=150)
    plt.close()

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
