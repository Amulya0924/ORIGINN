import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import load_and_preprocess
from src.detect import detect_oil_slick


def create_overlay(original_display_array, mask, alpha=0.4):
    """
    Overlays a binary detection mask onto the original grayscale SAR display array
    using semi-transparent red highlight.

    Parameters:
        original_display_array (np.ndarray): Normalized 0-255 uint8 grayscale image array (H, W).
        mask (np.ndarray): Binary detection mask (H, W) with values 0 (background) and 255 (slick).
        alpha (float): Transparency factor for the red overlay (default: 0.4).

    Returns:
        np.ndarray: Composited RGB image array (H, W, 3) with uint8 data type.
    """
    if original_display_array.ndim == 2:
        rgb_img = cv2.cvtColor(original_display_array, cv2.COLOR_GRAY2RGB).astype(np.float32)
    else:
        rgb_img = original_display_array.copy().astype(np.float32)

    red_color = np.array([255.0, 0.0, 0.0], dtype=np.float32)

    slick_indices = (mask == 255)
    rgb_img[slick_indices] = (1.0 - alpha) * rgb_img[slick_indices] + alpha * red_color

    composited_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

    return composited_img


def create_error_map(pred_mask, gt_mask):
    """
    Generates a color-coded segmentation error map comparing predicted mask to ground truth mask.

    Colors:
        - Green ([0, 255, 0]): True Positive (Correctly detected oil slick)
        - Red ([255, 0, 0]): False Positive (False alarm - ocean detected as slick)
        - Blue ([0, 120, 255]): False Negative (Missed oil slick)
        - Dark Gray ([30, 30, 30]): True Negative (Correctly identified ocean)

    Parameters:
        pred_mask (np.ndarray): Predicted binary mask (0 or 255 uint8).
        gt_mask (np.ndarray): Ground truth binary mask (0 or 255 uint8).

    Returns:
        np.ndarray: Color-coded RGB image array (H, W, 3) with uint8 data type.
    """
    pred_b = (pred_mask > 127)
    gt_b = (gt_mask > 127)

    error_rgb = np.full((*pred_mask.shape, 3), fill_value=30, dtype=np.uint8)

    # True Positive: Green
    tp = np.logical_and(pred_b, gt_b)
    error_rgb[tp] = [0, 230, 115]

    # False Positive: Red
    fp = np.logical_and(pred_b, np.logical_not(gt_b))
    error_rgb[fp] = [235, 60, 60]

    # False Negative: Blue
    fn = np.logical_and(np.logical_not(pred_b), gt_b)
    error_rgb[fn] = [50, 150, 255]

    return error_rgb


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    output_dir = Path("data/output")

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_input_path = raw_dir / "dataset" / "images" / "val" / "palsar_0.png"

    if not sample_input_path.exists():
        print(f"Sample input {sample_input_path} not found.")
        sys.exit(1)

    print("--- Full Pipeline End-to-End Execution on Real SAR Image ---")
    print(f"Step 1: Loading and preprocessing {sample_input_path}...")
    db_array, norm_array = load_and_preprocess(sample_input_path)

    print("Step 2: Detecting oil slick candidate regions...")
    slick_mask = detect_oil_slick(db_array)

    print("Step 3: Creating semi-transparent red overlay visualization...")
    overlay_img = create_overlay(norm_array, slick_mask, alpha=0.4)

    output_comparison_path = output_dir / "sample_vv_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(norm_array, cmap="gray")
    axes[0].set_title("Preprocessed Real SAR Image (VV dB)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title("Oil Slick Detection Overlay", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_comparison_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Pipeline finished successfully!")
    print(f"Saved side-by-side comparison PNG to: {output_comparison_path}")
