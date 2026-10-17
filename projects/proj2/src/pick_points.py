"""Run this yourself in a terminal with a display (not through Claude) to
click 2 corresponding points on each of two images. Prints the coordinates
so you can paste them back.

Usage: uv run python src/pick_points.py <image1> <image2>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "hybrid_starter"))
from align_image_code import get_points  # noqa: E402


def main():
    im1 = plt.imread(sys.argv[1])
    im2 = plt.imread(sys.argv[2])
    p1, p2, p3, p4 = get_points(im1, im2)
    print(f"p1, p2 = {tuple(p1)}, {tuple(p2)}")
    print(f"p3, p4 = {tuple(p3)}, {tuple(p4)}")


if __name__ == "__main__":
    main()
