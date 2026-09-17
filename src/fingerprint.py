"""
Geometric feature extraction module for SAR oil slick detection masks.
Computes real spatial and morphological properties (area, centroid, bounding box, orientation, elongation).
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_fingerprint(mask, confidence_val=None):
    """
    Computes real spatial geometric properties from a binary predicted oil slick mask using OpenCV.

    Parameters:
        mask (np.ndarray): Binary detection mask (0 or 255 uint8).
        confidence_val (float, optional): Raw model confidence score if available.

    Returns:
        dict: Geometric fingerprint parameters:
            - area_pixels (int): Count of detected slick pixels.
            - area_pct (float): Percentage of total raster scene area.
            - centroid (tuple): (cx, cy) spatial centroid coordinates.
            - length_px (float): Length of minimum area bounding box in pixels.
            - width_px (float): Width of minimum area bounding box in pixels.
            - angle_deg (float): Orientation angle of minimum area bounding box in degrees.
            - elongation_ratio (float): Ratio of length to width.
            - shape_classification (str): "Elongated" or "Compact".
            - raw_confidence (float or None): Raw detection confidence value if present.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Invalid mask input provided.")

    total_pixels = mask.size
    slick_pixels = np.count_nonzero(mask == 255)
    area_pct = float((slick_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    # If no slick pixels detected, return zeroed fingerprint
    if slick_pixels == 0:
        return {
            "area_pixels": 0,
            "area_pct": 0.0,
            "centroid": (0.0, 0.0),
            "length_px": 0.0,
            "width_px": 0.0,
            "angle_deg": 0.0,
            "elongation_ratio": 1.0,
            "shape_classification": "None (No Slick Detected)",
            "raw_confidence": confidence_val
        }

    # 1. Centroid calculation using spatial moments (cv2.moments)
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx, cy = 0.0, 0.0

    # 2. Minimum Area Bounding Box, Length, Width, and Orientation Angle (cv2.minAreaRect)
    non_zero_pts = cv2.findNonZero(mask)
    if non_zero_pts is not None:
        rect = cv2.minAreaRect(non_zero_pts)
        (center_x, center_y), (dim1, dim2), angle = rect
        length = max(dim1, dim2)
        width = min(dim1, dim2)
    else:
        length, width, angle = 0.0, 0.0, 0.0

    # 3. Elongation ratio & shape classification
    if width > 0:
        elongation_ratio = float(length / width)
    else:
        elongation_ratio = 1.0

    shape_classification = "Elongated" if elongation_ratio >= 1.8 else "Compact"

    fingerprint = {
        "area_pixels": int(slick_pixels),
        "area_pct": round(area_pct, 2),
        "centroid": (round(cx, 1), round(cy, 1)),
        "length_px": round(float(length), 1),
        "width_px": round(float(width), 1),
        "angle_deg": round(float(angle), 1),
        "elongation_ratio": round(elongation_ratio, 2),
        "shape_classification": shape_classification,
        "raw_confidence": confidence_val
    }

    return fingerprint


if __name__ == "__main__":
    # Test fingerprint extraction on a synthetic binary slick mask
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(test_mask, (128, 128), (60, 20), 30, 0, 360, 255, -1)

    fp = extract_fingerprint(test_mask)
    print("Extracted Test Fingerprint:")
    for k, v in fp.items():
        print(f"  {k}: {v}")
