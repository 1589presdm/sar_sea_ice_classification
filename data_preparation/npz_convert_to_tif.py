"""
Convert segmentation mask from npz format to TIFF files.

Expected labels in the npz mask:
0 = ignored area
1 = water 
2 = ice 

Output TIFF labels:
-1 = ignored area
0 = ice 
1 = water 

"""


from pathlib import Path
import numpy as np
import rasterio

npz_dir = Path("") # directory containing npz files 
mask_out_dir = Path("") # directory where converted mask will be saved 
mask_out_dir.mkdir(parents=True, exist_ok=True)
input_suffix = "_ice_mask.npz" # change if source files use a different suffix

expected_labels = {0, 1, 2}
  
for npz_path in sorted(npz_dir.glob(f"*{input_suffix}")): 
    with np.load(npz_path) as data:
        original_mask = data["ice_mask"].astype(np.int16)

    found_labels = set(np.unique(original_mask))

    unexpected = found_labels - expected_labels
    if unexpected:
        raise ValueError(f"unexpected labels: {unexpected}")

    mask = np.empty_like(original_mask, dtype=np.int16)

    # remap source labels to training labels
    mask[original_mask == 0] = -1   # ignore
    mask[original_mask == 1] = 1    # water
    mask[original_mask == 2] = 0    # ice


    out_name = npz_path.name.removesuffix(input_suffix) + "_mask.tif"
    out_path = mask_out_dir / out_name

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype="int16",
    ) as dst:
        dst.write(mask, 1)

    print(
        f"saved: {out_path.name} | "
        f"original_mask={np.unique(original_mask, return_counts=True)} | "
        f"output_mask={np.unique(mask, return_counts=True)}"
    )