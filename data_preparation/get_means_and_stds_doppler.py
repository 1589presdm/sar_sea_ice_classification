"""
Compute mean and standart deveation for pseudo doppler TIFF files.
Metrics are computed only from the train.txt from splits.

"""

from pathlib import Path
import numpy as np
import tifffile

doppler_dir = Path("")  # path to doppler TIFF files
train_split_path = Path("")  # path to train.txt file

train_ids = [line.strip() for line in train_split_path.read_text().splitlines() if line.strip()]

sum_pixels = 0.0
sum_sq_pixels = 0.0
num_pixels = 0
used_images = 0

for sample_id in train_ids:
    tif_path = doppler_dir / f"{sample_id}_doppler.tif"

    if not tif_path.exists():
        print(f"file not found: {tif_path}")
        continue
    
    arr = tifffile.imread(tif_path)
    
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    
    if arr.ndim != 2:
        print(f"unexpected shape for {tif_path.name}: {arr.shape}")
        continue
    
    arr = arr.astype(np.float64)
    
    sum_pixels += arr.sum()
    sum_sq_pixels += (arr ** 2).sum()
    num_pixels += arr.size
    used_images += 1
    
mean = sum_pixels / num_pixels
variance = (sum_sq_pixels / num_pixels) - (mean ** 2)
std = np.sqrt(variance)

print("train images used:", used_images)
print("means:", [float(mean)])
print("stds:", [float(std)])