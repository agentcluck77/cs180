"""Part 2.2: Hybrid Images.

hybrid = high-pass(im1, sigma1) + low-pass(im2, sigma2)

Uses the provided align_image_code.py. Correspondence points were picked by
you, running src/pick_points.py locally (plt.ginput() needs a display),
then hardcoded here.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import DATA, OUT, gaussian_blur, to_gray, load_color, save_color, clip_uint8

sys.path.insert(0, str(DATA / "hybrid_starter"))
from align_image_code import align_image_centers, rescale_images, rotate_im1, match_img_size  # noqa: E402


def align_with_points(im1, im2, p1, p2, p3, p4):
    """Aligns im1, im2, and also tracks which pixels are original image data
    vs. synthetic padding by carrying an all-ones "coverage" channel through
    the same transforms (can't tell padding from real dark content, e.g. a
    tiger's black stripes, by pixel darkness alone)."""
    im1x = np.dstack([im1, np.ones(im1.shape[:2])])
    im2x = np.dstack([im2, np.ones(im2.shape[:2])])

    pts = (p1, p2, p3, p4)
    im1x, im2x = align_image_centers(im1x, im2x, pts)
    im1x, im2x = rescale_images(im1x, im2x, pts)
    im1x, _angle = rotate_im1(im1x, pts)
    im1x, im2x = match_img_size(im1x, im2x)

    im1a, cover1 = im1x[..., :3], im1x[..., 3]
    im2a, cover2 = im2x[..., :3], im2x[..., 3]
    return im1a, im2a, cover1, cover2


def crop_to_valid_center(im1a, im2a, cover1, cover2, thresh=0.999):
    """Shrink a box symmetric about the center until it's ~thresh covered by
    real image data (not padding) in both images."""
    H, W = im1a.shape[:2]
    valid = (cover1 > 0.999) & (cover2 > 0.999)
    cy, cx = H // 2, W // 2

    for m in np.arange(1.0, 0.0, -0.01):
        hh, hw = int(H / 2 * m), int(W / 2 * m)
        box = valid[cy - hh : cy + hh, cx - hw : cx + hw]
        if box.mean() >= thresh:
            return im1a[cy - hh : cy + hh, cx - hw : cx + hw], im2a[cy - hh : cy + hh, cx - hw : cx + hw]
    return im1a, im2a


def hybrid_image(im1: np.ndarray, im2: np.ndarray, sigma1: float, sigma2: float) -> np.ndarray:
    """im1 contributes high frequencies, im2 contributes low frequencies."""
    high = im1 - gaussian_blur(im1, sigma1)
    low = gaussian_blur(im2, sigma2)
    return high + low


def fft_log_mag(gray: np.ndarray) -> np.ndarray:
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def save_fft(path, gray: np.ndarray):
    fm = fft_log_mag(gray)
    plt.imsave(path, fm, cmap="gray")


def make_hybrid_example(name, im1, im2, p1, p2, p3, p4, sigma1, sigma2, do_fft_analysis=False):
    im1a, im2a, cover1, cover2 = align_with_points(im1, im2, p1, p2, p3, p4)
    im1a, im2a = crop_to_valid_center(im1a, im2a, cover1, cover2)

    g1, g2 = to_gray(im1a), to_gray(im2a)
    hybrid_gray = hybrid_image(g1, g2, sigma1, sigma2)

    save_color(OUT / f"2_2_{name}_im1_aligned.jpg", im1a)
    save_color(OUT / f"2_2_{name}_im2_aligned.jpg", im2a)
    cv2.imwrite(str(OUT / f"2_2_{name}_hybrid_gray.jpg"), clip_uint8(hybrid_gray))

    if do_fft_analysis:
        low2 = gaussian_blur(g2, sigma2)
        high1 = g1 - gaussian_blur(g1, sigma1)
        save_fft(OUT / f"2_2_{name}_fft_im1.png", g1)
        save_fft(OUT / f"2_2_{name}_fft_im2.png", g2)
        save_fft(OUT / f"2_2_{name}_fft_im1_highpass.png", high1)
        save_fft(OUT / f"2_2_{name}_fft_im2_lowpass.png", low2)
        save_fft(OUT / f"2_2_{name}_fft_hybrid.png", hybrid_gray)

    print(f"[{name}] aligned shape={im1a.shape}, sigma1={sigma1}, sigma2={sigma2}")
    return im1a, im2a, hybrid_gray


def color_hybrid_variants(name, im1a, im2a, sigma1, sigma2):
    """Bells & whistles: does color help the high-freq image, the low-freq
    image, or both?"""
    g1, g2 = to_gray(im1a), to_gray(im2a)

    color_high = im1a - gaussian_blur(im1a, sigma1)
    color_low = gaussian_blur(im2a, sigma2)
    gray_high = (g1 - gaussian_blur(g1, sigma1))[..., None]
    gray_low = gaussian_blur(g2, sigma2)[..., None]

    save_color(OUT / f"2_2_{name}_hybrid_color_both.jpg", color_high + color_low)
    save_color(OUT / f"2_2_{name}_hybrid_color_low_only.jpg", gray_high + color_low)
    save_color(OUT / f"2_2_{name}_hybrid_color_high_only.jpg", color_high + gray_low)


def main():
    derek = load_color(DATA / "hybrid_starter/DerekPicture.jpg")
    nutmeg = load_color(DATA / "hybrid_starter/nutmeg.jpg")

    # Eye coordinates from your own plt.ginput() clicks (src/pick_points.py).
    p1, p2 = (297.62, 351.08), (441.69, 335.85)   # Derek: left eye, right eye
    p3, p4 = (607.79, 289.21), (757.79, 370.64)   # Nutmeg: left eye, right eye

    im1a, im2a, _ = make_hybrid_example(
        "derek_nutmeg", derek, nutmeg, p1, p2, p3, p4,
        sigma1=8, sigma2=8, do_fft_analysis=True,
    )
    color_hybrid_variants("derek_nutmeg", im1a, im2a, sigma1=8, sigma2=8)

    cat = load_color(DATA / "stock/cat_face.jpg")
    tiger = load_color(DATA / "stock/tiger_face.jpg")

    # Eye coordinates from your own plt.ginput() clicks.
    q1, q2 = (1031.51, 702.32), (1369.46, 706.86)   # cat: left eye, right eye
    q3, q4 = (442.27, 497.55), (737.72, 508.92)     # tiger: left eye, right eye

    make_hybrid_example(
        "cat_tiger", cat, tiger, q1, q2, q3, q4,
        sigma1=18, sigma2=18, do_fft_analysis=True,
    )

    basketball = load_color(DATA / "stock/basketball.jpg")
    earth = load_color(DATA / "stock/earth.jpg")

    # Opposite-edge (diameter) points from your own plt.ginput() clicks.
    r1, r2 = (54.04, 378.23), (771.05, 378.23)      # basketball: left edge, right edge
    r3, r4 = (228.35, 1010.20), (1854.67, 1062.84)  # earth: left edge, right edge

    make_hybrid_example(
        "basketball_earth", basketball, earth, r1, r2, r3, r4,
        sigma1=20, sigma2=14, do_fft_analysis=True,
    )

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
