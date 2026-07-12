"""
Generate pseudodoppler TIFF from PNG images.

"""

from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import gaussian_filter

png_dir = Path("") #path to the directory of png files
doppler_out_dir = Path("") # path to the directory for saving converted files 
doppler_out_dir.mkdir(parents=True, exist_ok=True)

sigma = 2.0 # gaussian smoothing intensity(higher values produce smoother results)
downsample_factor = 4 #factpr used to downsample the smoothed map before resizing it back 

def decode_png_channel(channel_uint8: np.ndarray) -> np.ndarray:
    channel = channel_uint8.astype(np.float32)
    decoded = np.full(channel.shape, np.nan, dtype=np.float32)

    valid = channel > 0
    decoded[valid] = (channel[valid] - 1.0) / 254.0

    return np.clip(decoded, 0.0, 1.0)

for png_path in sorted(png_dir.glob("*.png")):
    arr = np.array(Image.open(png_path).convert("RGB"))

    if arr.ndim != 3 or arr.shape[2] != 3:
        print(f"unexpected shape {arr.shape} in {png_path.name}")
        continue
    
    hv_png = arr[:, :, 0]  # R = HV
    hh_png = arr[:, :, 2]  # B = HH
    
    hv = decode_png_channel(hv_png)
    hh = decode_png_channel(hh_png)

    # pseudodoppler formula derived from HH and HV channels
    doppler_proxy = hv * (hv + 2.0 * hh * (1.0 - hv))
    doppler_proxy = np.clip(doppler_proxy, 0.0, 1.0)
    
    doppler_smooth = gaussian_filter(doppler_proxy, sigma=sigma)
    
    height, width = doppler_smooth.shape
    small_height = max(1, height // downsample_factor)
    small_width = max(1, width // downsample_factor)
    
    # downsample and restore for lower resolution structure imitation 
    doppler_image = Image.fromarray((doppler_smooth * 255).astype(np.uint8))
    doppler_small = doppler_image.resize(
        (small_width, small_height),
        resample=Image.BILINEAR,
    )
    doppler_restored = doppler_small.resize(
        (width, height),
        resample=Image.BILINEAR,
    )
    
    doppler_final = np.array(doppler_restored).astype(np.float32) / 255.0
    doppler = doppler_final[np.newaxis, :, :].astype(np.float32)
    
    out_path = doppler_out_dir / f"{png_path.stem}_doppler.tif"
    tifffile.imwrite(out_path, doppler)
    
    print(
        f"saved: {out_path.name} | shape={doppler.shape} | dtype={doppler.dtype} "
        f"min={np.nanmin(doppler):.4f} | max={np.nanmax(doppler):.4f}"
    )