"""
Create train/val/test splits. 

Files are grouped by schene ID, before the # symbol.
All pathces from the same scene placed in the same split.

For example:
scene_001#0_0.tif
scene_001#0_1.tif
scene_001#1_0.tif
scene_001#1_1.tif

All these patches will save to one split.
"""

from pathlib import Path
from collections import defaultdict
import random

images_dir = Path("") # path to tif images
splits_dir = Path("")  #path where splits data will be saved 
splits_dir.mkdir(parents=True, exist_ok=True)

train_ratio = 0.70
val_ratio = 0.15

files = sorted([path.stem for path in images_dir.glob("*.tif")])

groups = defaultdict(list)

for file_id in files:
    scene_id = file_id.split("#")[0]
    groups[scene_id].append(file_id)
    
scenes = sorted(groups.keys())

num_scenes = len(scenes)

train_end = int(train_ratio * num_scenes)
val_end = int((train_ratio + val_ratio) * num_scenes)

train_scenes = scenes[:train_end]
val_scenes = scenes[train_end:val_end]
test_scenes = scenes[val_end:]

def collect_patches(scene_list):
    patch_ids = []

    for scene_id in scene_list:
        patch_ids.extend(groups[scene_id])

    return sorted(patch_ids)

train = collect_patches(train_scenes)
val = collect_patches(val_scenes)
test = collect_patches(test_scenes)

(splits_dir / "train.txt").write_text("\n".join(train) + "\n")
(splits_dir / "val.txt").write_text("\n".join(val) + "\n")
(splits_dir / "test.txt").write_text("\n".join(test) + "\n")

print(f"scenes total: {num_scenes}")
print(f"train: {len(train)} patches, {len(train_scenes)} scenes")
print(f"val: {len(val)} patches, {len(val_scenes)} scenes")
print(f"test: {len(test)} patches, {len(test_scenes)} scenes")