"""
Evaluation module for comparing predicted oil slick segmentation masks against ground truth masks.
"""

import numpy as np


def calculate_metrics(pred_mask, gt_mask):
    """
    Computes segmentation performance metrics (IoU, Dice/F1 Score, Precision, Recall, Accuracy)
    between a predicted binary mask and a ground truth binary mask.

    Parameters:
        pred_mask (np.ndarray): Binary prediction mask (values 0 or 255).
        gt_mask (np.ndarray): Binary ground truth mask (values 0 or 255).

    Returns:
        dict: Dictionary containing calculated metrics:
            - iou (float): Intersection over Union (Jaccard Index)
            - dice (float): Dice Similarity Coefficient / F1 Score
            - precision (float): Precision of oil slick detection
            - recall (float): Recall / Sensitivity of oil slick detection
            - accuracy (float): Overall pixel classification accuracy
    """
    # Convert masks to boolean arrays (True for slick pixels)
    pred_binary = (pred_mask > 127)
    gt_binary = (gt_mask > 127)

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()

    iou = float(intersection / union) if union > 0 else (1.0 if pred_sum == gt_sum == 0 else 0.0)
    dice = float(2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else (1.0 if pred_sum == gt_sum == 0 else 0.0)
    precision = float(intersection / pred_sum) if pred_sum > 0 else (1.0 if gt_sum == 0 else 0.0)
    recall = float(intersection / gt_sum) if gt_sum > 0 else (1.0 if pred_sum == 0 else 0.0)

    total_pixels = pred_mask.size
    correct_pixels = (pred_binary == gt_binary).sum()
    accuracy = float(correct_pixels / total_pixels)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Test evaluation module with synthetic mask data
    pred = np.zeros((100, 100), dtype=np.uint8)
    gt = np.zeros((100, 100), dtype=np.uint8)

    pred[20:60, 20:60] = 255
    gt[25:65, 25:65] = 255

    results = calculate_metrics(pred, gt)
    print("Test Evaluation Metrics:")
    for k, v in results.items():
        print(f"  {k.upper()}: {v:.4f}")
