import os
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import median_filter
from PIL import Image


def load_and_preprocess(filepath):
    """
    Loads a SAR imagery file (GeoTIFF .tif/.tiff or image format .png/.jpg), converts amplitude/intensity
    to decibels (dB), applies median filtering to reduce speckle noise, and normalizes for display.

    Parameters:
        filepath (str or Path): Path to the input SAR raster or image file.

    Returns:
        tuple: (db_array, norm_array)
            - db_array (np.ndarray): Un-normalized dB backscatter array (float32).
            - norm_array (np.ndarray): Filtered and 0-255 uint8 normalized image array for display.
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".tif", ".tiff"]:
        with rasterio.open(filepath) as src:
            img = src.read(1).astype(np.float32)
    else:
        # Load standard image formats (PNG, JPG) and convert to float grayscale intensity
        pil_img = Image.open(filepath).convert("L")
        img = np.array(pil_img, dtype=np.float32)

    # 1. Convert amplitude/intensity to decibels (10 * log10(x)) safely
    epsilon = 1e-10
    safe_img = np.maximum(img, epsilon)
    db_array = 10.0 * np.log10(safe_img)
    db_array = np.nan_to_num(db_array, nan=-100.0, posinf=0.0, neginf=-100.0)

    # 2. Apply median filter (size=5) to reduce speckle noise
    filtered_db = median_filter(db_array, size=5)

    # 3. Normalize result to 0-255 uint8 range for display
    min_val = np.min(filtered_db)
    max_val = np.max(filtered_db)

    if max_val > min_val:
        norm_array = ((filtered_db - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        norm_array = np.zeros_like(filtered_db, dtype=np.uint8)

    return db_array, norm_array


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    processed_dir.mkdir(parents=True, exist_ok=True)

    sample_input_path = raw_dir / "dataset" / "images" / "val" / "palsar_0.png"

    if not sample_input_path.exists():
        print(f"Dataset file {sample_input_path} not found.")
        exit(1)

    print(f"Loading and preprocessing real SAR image: {sample_input_path}...")
    db_array, norm_array = load_and_preprocess(sample_input_path)

    output_png_path = processed_dir / "palsar_0_preprocessed.png"
    output_image = Image.fromarray(norm_array)
    output_image.save(output_png_path)

    print(f"Successfully processed real SAR image!")
    print(f"Decibel (dB) Array Shape: {db_array.shape}, Min: {db_array.min():.2f} dB, Max: {db_array.max():.2f} dB")
    print(f"Normalized Display Array Shape: {norm_array.shape}, Range: [{norm_array.min()}, {norm_array.max()}]")
    print(f"Saved preprocessed visualization to: {output_png_path}")
