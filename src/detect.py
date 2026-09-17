import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import load_and_preprocess


def detect_oil_slick(db_array, threshold_offset=0):
    """
    Detects low-backscatter (dark) oil-slick candidate regions from a SAR dB array using
    Otsu's thresholding followed by morphological filtering.

    Parameters:
        db_array (np.ndarray): Decibel backscatter array or 0-255 normalized image array.
        threshold_offset (int): Offset added to Otsu's automatic threshold to adjust sensitivity.

    Returns:
        np.ndarray: Binary mask of detected oil slick regions (uint8 array with values 0 or 255).
    """
    if db_array.dtype != np.uint8:
        min_val = np.min(db_array)
        max_val = np.max(db_array)
        if max_val > min_val:
            img_uint8 = ((db_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.zeros_like(db_array, dtype=np.uint8)
    else:
        img_uint8 = db_array

    # 1. Apply Otsu's thresholding to determine base threshold
    otsu_thresh, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply offset for user sensitivity adjustment
    effective_thresh = int(np.clip(otsu_thresh + threshold_offset, 0, 255))
    _, binary_mask = cv2.threshold(img_uint8, effective_thresh, 255, cv2.THRESH_BINARY_INV)

    # 2. Apply morphological opening then closing with a 5x5 kernel to remove noise
    kernel = np.ones((5, 5), dtype=np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    output_dir = Path("data/output")

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_input_path = raw_dir / "dataset" / "images" / "val" / "palsar_0.png"

    if not sample_input_path.exists():
        print(f"Sample input {sample_input_path} not found.")
        sys.exit(1)

    print(f"Loading and preprocessing real SAR image: {sample_input_path}...")
    db_array, _ = load_and_preprocess(sample_input_path)

    print("Running oil slick detection...")
    slick_mask = detect_oil_slick(db_array)

    output_mask_path = output_dir / "palsar_0_slick_mask.png"
    mask_image = Image.fromarray(slick_mask)
    mask_image.save(output_mask_path)

    detected_pixels = np.count_nonzero(slick_mask == 255)
    total_pixels = slick_mask.size
    print(f"Detection completed successfully!")
    print(f"Mask Shape: {slick_mask.shape}, Unique Values: {np.unique(slick_mask)}")
    print(f"Detected Oil Slick Pixels: {detected_pixels} / {total_pixels} ({detected_pixels / total_pixels * 100:.2f}%)")
    print(f"Saved binary detection mask to: {output_mask_path}")
