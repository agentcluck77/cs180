"""Part 2.4: Multiresolution Blending (the oraple, and 2 blends of our own)."""

import cv2
import numpy as np

from stacks import gaussian_stack, laplacian_stack
from utils import DATA, OUT, clip_uint8, load_color, save_color, to_gray

LEVELS = 6
SIGMA = 8


def blend(im1: np.ndarray, im2: np.ndarray, mask: np.ndarray):
    """mask is 1 where im1 shows through, 0 where im2 shows through."""
    g1, g2, gm = gaussian_stack(im1, LEVELS, SIGMA), gaussian_stack(im2, LEVELS, SIGMA), gaussian_stack(mask, LEVELS, SIGMA)
    l1, l2 = laplacian_stack(g1), laplacian_stack(g2)
    blended_levels = [gm[i] * l1[i] + (1 - gm[i]) * l2[i] for i in range(LEVELS)]
    return sum(blended_levels), blended_levels, l1, l2


def vertical_step_mask(shape):
    H, W = shape[:2]
    mask = np.zeros((H, W))
    mask[:, : W // 2] = 1.0
    return mask[..., None]


def center_square_crop(img, cx=None, cy=None, half=None):
    H, W = img.shape[:2]
    if cx is None:
        cx, cy = W // 2, H // 2
    if half is None:
        half = min(cx, cy, W - cx, H - cy)
    return img[cy - half : cy + half, cx - half : cx + half]


def save_process_figure(name, im1, im2, mask, l1, l2, result):
    """Figure 10 (Burt & Adelson) style: masked inputs + a couple of
    Laplacian levels + the final blend."""
    import matplotlib.pyplot as plt

    def norm01(im):
        lo, hi = im.min(), im.max()
        return (im - lo) / (hi - lo + 1e-12)

    show_levels = [0, 2, LEVELS - 1]
    fig, axes = plt.subplots(2, 2 + len(show_levels), figsize=(4 * (2 + len(show_levels)), 8))
    axes[0, 0].imshow(np.clip(im1, 0, 1)); axes[0, 0].set_title("input 1")
    axes[0, 1].imshow(mask.squeeze(), cmap="gray"); axes[0, 1].set_title("mask")
    axes[1, 0].imshow(np.clip(im2, 0, 1)); axes[1, 0].set_title("input 2")
    axes[1, 1].imshow(np.clip(result, 0, 1)); axes[1, 1].set_title("blended result")
    for i, lvl in enumerate(show_levels):
        axes[0, 2 + i].imshow(norm01(l1[lvl])); axes[0, 2 + i].set_title(f"Laplacian 1, level {lvl}")
        axes[1, 2 + i].imshow(norm01(l2[lvl])); axes[1, 2 + i].set_title(f"Laplacian 2, level {lvl}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / f"2_4_{name}_process.png", dpi=150)
    plt.close()


def main():
    # --- Required: the oraple (apple/orange, vertical seam) ---
    apple = load_color(DATA / "apple_orange/apple.jpeg")
    orange = load_color(DATA / "apple_orange/orange.jpeg")
    mask = vertical_step_mask(apple.shape)
    result, _, l1, l2 = blend(apple, orange, mask)
    save_color(OUT / "2_4_oraple_result.jpg", result)
    save_process_figure("oraple", apple, orange, mask, l1, l2, result)
    print("[oraple] done")

    # --- Bells & whistles: does color actually help here? ---
    apple_gray = np.repeat(to_gray(apple)[..., None], 3, axis=-1)
    orange_gray = np.repeat(to_gray(orange)[..., None], 3, axis=-1)
    result_gray, *_ = blend(apple_gray, orange_gray, mask)
    save_color(OUT / "2_4_oraple_result_grayscale.jpg", result_gray)
    print("[oraple grayscale comparison] done")

    # --- Custom pair 1: lemon/lime, straight (vertical) seam ---
    lemon = load_color(DATA / "stock/lemon.jpg")
    lime = load_color(DATA / "stock/lime.jpg")
    size = 500
    # Plain center crops put the two fruits at very different apparent scales
    # (the lemon photo has margin around it, lime_raw.jpg is a wide shot with a
    # second halved lime in frame). Both crops below are sized so the fruit's own
    # body fills a similar fraction of the frame, measured by hand from a grid
    # overlay on each source image.
    lemon_sq = cv2.resize(center_square_crop(lemon, cx=1281, cy=1020, half=1020), (size, size))
    lime_sq = cv2.resize(center_square_crop(lime, cx=2306, cy=2439, half=1809), (size, size))
    mask2 = vertical_step_mask(lemon_sq.shape)
    result2, _, l1b, l2b = blend(lemon_sq, lime_sq, mask2)
    save_color(OUT / "2_4_lemonlime_result.jpg", result2)
    save_process_figure("lemonlime", lemon_sq, lime_sq, mask2, l1b, l2b, result2)
    print("[lemonlime] done")

    # --- Custom pair 2: eye/moon, irregular (circular) mask over the pupil ---
    eye = load_color(DATA / "stock/eye.jpg")
    moon = load_color(DATA / "stock/moon.jpeg")
    canvas = 900
    # eye photo cropped/centered on the pupil (detected via contour thresholding)
    eye_sq = cv2.resize(center_square_crop(eye, cx=854, cy=900, half=700), (canvas, canvas))
    pupil_radius = int(280 * canvas / 1400)

    moon_small_size = pupil_radius * 2
    moon_small = cv2.resize(moon, (moon_small_size, moon_small_size))
    moon_layer = np.zeros((canvas, canvas, 3))
    off = (canvas - moon_small_size) // 2
    moon_layer[off : off + moon_small_size, off : off + moon_small_size] = moon_small

    yy, xx = np.mgrid[0:canvas, 0:canvas]
    dist = np.sqrt((xx - canvas / 2) ** 2 + (yy - canvas / 2) ** 2)
    circ_mask = (dist > pupil_radius).astype(np.float64)[..., None]  # 1 = eye, 0 = moon

    result3, _, l1c, l2c = blend(eye_sq, moon_layer, circ_mask)
    save_color(OUT / "2_4_eyemoon_result.jpg", result3)
    save_process_figure("eyemoon", eye_sq, moon_layer, circ_mask, l1c, l2c, result3)
    print("[eyemoon] done")

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
