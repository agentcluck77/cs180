"""CS180 Project 1 - Prokudin-Gorskii colorization."""

import os
import sys

import cv2
import numpy as np

if len(sys.argv) < 3:
    print("error: input and output image paths are required")
    print("usage: uv run proj1final.py <input_image> <output_image>")
    sys.exit(1)

in_path = sys.argv[1]
out_path = sys.argv[2]

img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError(f"cv2 could not read {in_path}")

# just to test image reading.
print(img.shape)
#
# cv2.imshow("plate", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

h = img.shape[0] // 3

# slice out the 3 channels

b = img[0:h]
g = img[h : 2 * h]
r = img[2 * h : 3 * h]

# print(b.shape, g.shape, r.shape)


def crop_border(img, frac=0.1):
    dy = int(img.shape[0] * frac)
    dx = int(img.shape[1] * frac)
    return img[dy:-dy, dx:-dx]


# print(b.shape, crop_border(b).shape)


def ssd(a, b):
    a, b = crop_border(a), crop_border(b)
    diff = a.astype(float) - b.astype(float)
    return float((diff**2).sum())  # lower = better match


def ncc(a, b):
    a, b = crop_border(a), crop_border(b)
    av = a.astype(float).ravel()
    bv = b.astype(float).ravel()
    av -= av.mean()
    bv -= bv.mean()
    return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv)))  # higher = better


def align(target_channel, ref_channel, radius=15, center=(0, 0), metric="ssd"):
    score_fn = ssd if metric == "ssd" else ncc
    better = (
        (lambda new, best: new < best)
        if metric == "ssd"
        else (lambda new, best: new > best)
    )

    best_score = None
    best_shift = center
    cy, cx = center
    for dy in range(cy - radius, cy + radius + 1):
        for dx in range(cx - radius, cx + radius + 1):
            shifted = np.roll(target_channel, (dy, dx), axis=(0, 1))
            score = score_fn(shifted, ref_channel)
            if best_score is None or better(score, best_score):
                best_score = score
                best_shift = (dy, dx)
    return best_shift


def to_uint8(channel):
    "Normalize a channel (uint8 or uint16) to 0-255 uint8."
    if channel.dtype == np.uint8:
        return channel
    return (channel.astype(np.float32) / channel.max() * 255).astype(np.uint8)


# === BELLS & WHISTLE: Better Features ==========================
# Align on edge strength (image gradients) instead of raw pixel brightness.
# Raw SSD/NCC on brightness fails when the B/G/R plates were exposed
# differently. edges appear more stable across channels.
def gradient_mag(channel):
    "Edge strength instead of raw brightness - robust when channels differ in brightness."
    img8 = (channel.astype(np.float32) / channel.max() * 255).astype(np.uint8)
    gx = cv2.Sobel(img8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


# =================================================================


def downscale(img, scale=0.5):
    "Helper to shrink massive .tif images"
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def align_pyramid(
    target_channel,
    ref_channel,
    scale=0.5,
    min_size=400,
    refine_radius=2,
    base_radius=30,
    metric="ssd",
):
    if min(target_channel.shape) < min_size:
        return align(target_channel, ref_channel, radius=base_radius, metric=metric)

    small_target = downscale(target_channel, scale)
    small_ref = downscale(ref_channel, scale)
    coarse_dy, coarse_dx = align_pyramid(
        small_target, small_ref, scale, min_size, refine_radius, base_radius, metric
    )

    guess = (round(coarse_dy / scale), round(coarse_dx / scale))
    # the guess may be slightly off after scaling up, so search a
    # SMALL window around it at this resolution
    return align(
        target_channel, ref_channel, radius=refine_radius, center=guess, metric=metric
    )


# === BELLS & WHISTLE: Automatic Cropping =========================
# Detect the border instead of chopping off a fixed margin.
# real photo content has texture (high gradient energy), while the plate's border
# is close to a flat solid color (low gradient energy).
# Scan inward from each edge until gradient energy exceeds a small
# threshold, capped at max_frac of the image so it can never crop the photo itself.
def auto_crop(colour_img, max_frac=0.15):
    gray = cv2.cvtColor(colour_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx**2 + gy**2)
    row_energy = edge.mean(axis=1)
    col_energy = edge.mean(axis=0)
    row_thresh = row_energy.max() * 0.1
    col_thresh = col_energy.max() * 0.1

    h, w = gray.shape
    max_dy, max_dx = int(h * max_frac), int(w * max_frac)

    top = 0
    while top < max_dy and row_energy[top] < row_thresh:
        top += 1
    bottom = h
    while (h - bottom) < max_dy and row_energy[bottom - 1] < row_thresh:
        bottom -= 1
    left = 0
    while left < max_dx and col_energy[left] < col_thresh:
        left += 1
    right = w
    while (w - right) < max_dx and col_energy[right - 1] < col_thresh:
        right -= 1

    return colour_img[top:bottom, left:right]


# =================================================================


# === BELLS & WHISTLE: Automatic Contrasting =======================
# Linearly stretch intensities so the (estimated) darkest pixel
# maps to 0 and the brightest maps to 255. Uses the 0.5/99.5 percentiles
# so that dead/hot pixels won't affect this.


def auto_contrast(colour_img):
    img = colour_img.astype(np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = (img - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


# =================================================================


# === BELLS & WHISTLE: Automatic White Balance =====================
# Gray-world assumption: a "neutral" illuminant should make the average
# color of a scene roughly gray. Scale each channel so its mean matches
# the average of all three channel means.
def auto_white_balance(colour_img):
    img = colour_img.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)  # B, G, R
    gray_mean = channel_means.mean()
    scale = gray_mean / (channel_means + 1e-6)
    balanced = img * scale
    return np.clip(balanced, 0, 255).astype(np.uint8)


# =================================================================


# === BELLS & WHISTLE: Better Color Mapping =========================
# Prokudin-Gorskii's glass filters were not perfectly narrowband, so each
# plate picks up some "bleed" from its neighboring colors, muddying the
# result. This algo applies a simple decorrelation matrix that nudges each
# channel away from the average of the other two, boosting color
# separation. `strength` will control how aggressive the correction is.
def better_color_mapping(colour_img, strength=0.15):
    img = colour_img.astype(np.float32)
    m = np.array(
        [
            [1 + 2 * strength, -strength, -strength],
            [-strength, 1 + 2 * strength, -strength],
            [-strength, -strength, 1 + 2 * strength],
        ],
        dtype=np.float32,
    )
    out = (img.reshape(-1, 3) @ m.T).reshape(img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# =================================================================


# print(b.shape, downscale(b).shape)
# print("flat align:", align(g, b))
# print("pyramid:", align_pyramid(g, b))
# print("align around 0:", align(g, b))
# print("align around (5,2):", align(g, b, radius=3, center=(5, 2)))

# align on edge strength (gradient magnitude), not raw pixels - robust to
# brightness differences between channels (fixes cases like emir.tif)
gb, gg, gr = gradient_mag(b), gradient_mag(g), gradient_mag(r)
dg = align_pyramid(gg, gb, metric="ncc")
dr = align_pyramid(gr, gb, metric="ncc")
print(f"{in_path}: G offset {dg}, R offset {dr}")

# print("g shift", dg)
# print("r shift", dr)

g_aligned = np.roll(g, dg, axis=(0, 1))
r_aligned = np.roll(r, dr, axis=(0, 1))

colour = cv2.merge([to_uint8(b), to_uint8(g_aligned), to_uint8(r_aligned)])

cv2.imwrite(out_path, colour)
print(f"success! read {in_path}, wrote {out_path}")


# ===================================================================
# BELLS & WHISTLES DEMOS
# Each block below applies ONE bells-and-whistle to the aligned image and
# saves a before/after pair to output/bells_whistles/. These are separate
# from the main pipeline output above (out_path is untouched).
# ===================================================================
bw_dir = "output/bells_whistles"
os.makedirs(bw_dir, exist_ok=True)
stem = os.path.splitext(os.path.basename(in_path))[0]

# --- Better Features: raw-pixel alignment vs gradient-based alignment ---
raw_dg = align_pyramid(g, b, metric="ncc")
raw_dr = align_pyramid(r, b, metric="ncc")
raw_colour = cv2.merge(
    [
        to_uint8(b),
        to_uint8(np.roll(g, raw_dg, axis=(0, 1))),
        to_uint8(np.roll(r, raw_dr, axis=(0, 1))),
    ]
)
cv2.imwrite(f"{bw_dir}/{stem}_0_features_before_raw.jpg", raw_colour)
cv2.imwrite(f"{bw_dir}/{stem}_0_features_after_gradient.jpg", colour)
print(f"better features -> raw pixels: G{raw_dg} R{raw_dr}  gradients: G{dg} R{dr}")

# --- Automatic Cropping ---
cropped = auto_crop(colour)
cv2.imwrite(f"{bw_dir}/{stem}_1_crop_before.jpg", colour)
cv2.imwrite(f"{bw_dir}/{stem}_1_crop_after.jpg", cropped)

# --- Automatic Contrasting ---
contrasted = auto_contrast(cropped)
cv2.imwrite(f"{bw_dir}/{stem}_2_contrast_before.jpg", cropped)
cv2.imwrite(f"{bw_dir}/{stem}_2_contrast_after.jpg", contrasted)

# --- Automatic White Balance ---
balanced = auto_white_balance(contrasted)
cv2.imwrite(f"{bw_dir}/{stem}_3_whitebalance_before.jpg", contrasted)
cv2.imwrite(f"{bw_dir}/{stem}_3_whitebalance_after.jpg", balanced)

# --- Better Color Mapping ---
mapped = better_color_mapping(balanced)
cv2.imwrite(f"{bw_dir}/{stem}_4_colormap_before.jpg", balanced)
cv2.imwrite(f"{bw_dir}/{stem}_4_colormap_after.jpg", mapped)

cv2.imwrite(f"{bw_dir}/{stem}_5_final_all_bells.jpg", mapped)
print(f"bells & whistles demo images written to {bw_dir}/{stem}_*")
