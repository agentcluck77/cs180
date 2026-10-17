"""Part 2.1: Image "Sharpening" via unsharp masking.

sharpened = image + alpha * (image - blurred)
          = (1 + alpha) * image - alpha * blurred
which folds into a single convolution with kernel (1 + alpha) * impulse - alpha * G.
"""

import numpy as np

from utils import DATA, OUT, conv2d_per_channel, gaussian_kernel_2d, load_color, save_color


def unsharp_mask(img: np.ndarray, G: np.ndarray, alpha: float):
    blurred = conv2d_per_channel(img, G)
    high_freq = img - blurred
    sharpened = img + alpha * high_freq
    return blurred, high_freq, sharpened


def run_on_image(name: str, img: np.ndarray, ksize: int, sigma: float, alphas: list[float]):
    G = gaussian_kernel_2d(ksize, sigma)
    save_color(OUT / f"2_1_{name}_original.jpg", img)

    blurred, high_freq, _ = unsharp_mask(img, G, alphas[0])
    save_color(OUT / f"2_1_{name}_blurred.jpg", blurred)
    # high frequencies are signed and near-zero magnitude; renormalize to see them
    hf_vis = (high_freq - high_freq.min()) / (high_freq.max() - high_freq.min() + 1e-12)
    save_color(OUT / f"2_1_{name}_highfreq.jpg", hf_vis)

    for alpha in alphas:
        _, _, sharpened = unsharp_mask(img, G, alpha)
        save_color(OUT / f"2_1_{name}_sharpened_a{alpha:.1f}.jpg", sharpened)


def main():
    alphas = [0.5, 1.0, 2.0, 4.0]

    taj = load_color(DATA / "taj.jpg")
    run_on_image("taj", taj, ksize=9, sigma=3.0, alphas=alphas)

    portrait = load_color(DATA / "self_portrait.jpg")
    run_on_image("selfportrait", portrait, ksize=9, sigma=3.0, alphas=alphas)

    # --- Evaluation: blur a sharp image, then try to sharpen it back ---
    G_eval = gaussian_kernel_2d(9, 3.0)
    blurred_eval = conv2d_per_channel(portrait, G_eval)
    save_color(OUT / "2_1_eval_blurred.jpg", blurred_eval)

    _, _, resharpened = unsharp_mask(blurred_eval, G_eval, alpha=2.0)
    save_color(OUT / "2_1_eval_resharpened.jpg", resharpened)

    print(f"mean|original - blurred|      = {np.abs(portrait - blurred_eval).mean():.4f}")
    print(f"mean|original - resharpened|  = {np.abs(portrait - np.clip(resharpened, 0, 1)).mean():.4f}")
    print("(resharpening reduces but does not eliminate the gap: blurring destroys")
    print(" high-frequency info that unsharp masking can't truly recover, it only")
    print(" re-boosts whatever high-freq residual remains.)")

    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
