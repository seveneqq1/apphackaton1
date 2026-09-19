"""Create a printable ArUco marker for the ErgoNeck glasses mount."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
}


def create_marker(dictionary_name: str, marker_id: int, marker_pixels: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary_name])
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, marker_pixels)

    image = np.zeros((marker_pixels, marker_pixels), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_pixels, image, 1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a printable ArUco marker.")
    parser.add_argument("--id", type=int, default=0, help="Marker ID used by main.py (default: 0)")
    parser.add_argument("--dictionary", choices=DICTIONARIES, default="DICT_4X4_50")
    parser.add_argument("--pixels", type=int, default=1000, help="Black-square image width in pixels")
    parser.add_argument("--border", type=int, default=180, help="White border in pixels")
    parser.add_argument("--output", type=Path, default=Path("marker_0.png"))
    args = parser.parse_args()

    if args.pixels < 100 or args.border < 10:
        parser.error("--pixels must be >= 100 and --border must be >= 10")

    marker = create_marker(args.dictionary, args.id, args.pixels)
    printable = cv2.copyMakeBorder(
        marker, args.border, args.border, args.border, args.border, cv2.BORDER_CONSTANT, value=255
    )
    if not cv2.imwrite(str(args.output), printable):
        raise RuntimeError(f"Could not save {args.output}")
    print(f"Saved {args.output.resolve()}")
    print("Print at a size where the INNER black-and-white square is 4-5 cm wide.")


if __name__ == "__main__":
    main()
