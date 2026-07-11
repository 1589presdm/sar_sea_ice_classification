"""
Convert PNG images to two-band TIFF files.

Expected input PNG channels:
R = HV
G = ignored
B = HH

Output TIFF band:
Band1 = HH
Band2 = HV

"""
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio

png_dir = Path("") #path to the directory of png files 
s1_out_dir = Path("") #path to the directory for saving converted files 
s1_out_dir.mkdir(parents=True, exist_ok=True)

for png_path in sorted(png_dir.glob("*.png")):
    arr = np.array(Image.open(png_path).convert("RGB"))

    if arr.ndim != 3 or arr.shape[2] != 3:
        print(f"unexpected shape {arr.shape} in {png_path.name}")
        continue

    r = arr[:, :, 0]  # HV
    b = arr[:, :, 2]  # HH

    s1_stack = np.stack([b, r], axis=0).astype(np.int16)

    out_path = s1_out_dir / f"{png_path.stem}.tif"

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=s1_stack.shape[1],
        width=s1_stack.shape[2],
        count=2,
        dtype=s1_stack.dtype,
    ) as dst:
        dst.write(s1_stack)

    print(
    f"saved: {out_path.name} | shape={s1_stack.shape} | dtype={s1_stack.dtype} "
    f"| min={s1_stack.min()} | max={s1_stack.max()}"
)