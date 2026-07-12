"""
Compute  mean and standard deviation for two band TIFF files.

Expected bands:
Band1 = HH
Band2 = HV

Statistics are computed only from train.txt
"""

from pathlib import Path
import numpy as np
import tifffile

images_dir = Path("") # the path where are tif_images were saved 
train_split_path = Path("") # the path where splits files were saved

train_ids = [line.strip() for line in train_split_path.read_text().splitlines() if line.strip()]

sum_channels = None
sum_sq_channels = None
num_pixels = 0
expected_channels = 2
used_images = 0

for sample_id in train_ids:
    tif_path = images_dir / f"{sample_id}.tif"

    if not tif_path.exists():
        print(f"file not found: {tif_path}")
        continue

    arr = tifffile.imread(tif_path)

    if arr.ndim != 3:
        print(f"unexpected shape for {tif_path.name}: {arr.shape}")
        continue
    
    if arr.shape[0] == expected_channels:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[2] != expected_channels:
        print(f"unexpected number of channels for {tif_path.name}: {arr.shape}")
        continue
    
    arr = arr.astype(np.float64)
    
    pixels = arr.reshape(-1, arr.shape[2])

    valid = np.isfinite(pixels).all(axis=1)
    pixels = pixels[valid]
    
    if pixels.shape[0] == 0:
        print(f"no valid pixels in {tif_path.name}")
        continue
    
    if sum_channels is None:
        sum_channels = np.zeros(expected_channels, dtype=np.float64)
        sum_sq_channels = np.zeros(expected_channels, dtype=np.float64)
        
    sum_channels += pixels.sum(axis=0)
    sum_sq_channels += (pixels ** 2).sum(axis=0)
    num_pixels += pixels.shape[0]
    used_images += 1

means = sum_channels / num_pixels
variances = (sum_sq_channels / num_pixels) - (means ** 2)
stds = np.sqrt(variances)

print("train images used:", used_images)
print("means:", means.tolist())
print("stds:", stds.tolist())