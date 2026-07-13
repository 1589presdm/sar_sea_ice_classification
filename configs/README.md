### TerraTorch YAML Configuration 

This directory contains the YAML for training and evaluating the multimodal segmentation with TerraTorch CLI.

The YAML configuration contains:
```yaml 
custom_modules_path: ./thor_extension
```
The `thor_extension` directory is located in the repository root:

```text
sar_sea_ice_classification/
    configs/
    thor_extension/
        __init__.py
    pyproject.toml
```

The local `thor_extension` module works as a small wrapper that imports the installed extension and register THOR backbone and related components in TerraTorch.

Training commands should be executed from the root directory.

## Training

Before starting training update the dataset paths.

Run training:

```bash
uv run terratorch fit \
    --config configs/S1_multibackbone_multimodal.yaml
```

## Testing

Run testing:

```bash
uv run terratorch test \
    --config configs/S1_multibackbone_multimodal.yaml \
    --ckpt_path /path/to/checkoint001.ckpt
```

The checkpoint directory is defined in YAML configuration
`dirpath: ./results/thor_s1_doppler_yaml/checkpoints`

## Backbone patch size

The config uses two different backbone encoders:
```yaml
thor_v1_base
vit_small_patch16_224.dino
```

THOR uses flexible pathc size `flexivit_patch_size_seqs`.

The DINO backbone uses patch size 16. 

Because THOR and DINO use different patch size their features maps have different spatial resolution. 

For example, for crop size 288 x 288:
```text
THOR patch size 8 - 36 x 36 feature map
DINO pathc size 16 - 18 x 18 feature map
```
If the THOR patch size is changed, the feature map resolution aslo changes. In this case, `backbone_rescale_features` should be adjusted so that backbone feature maps can be merged correctly. 

`backbone_rescale_features` option controls how TerraTorch aligns these feature maps before concatenation. 

`backbone_rescale_features: up` upsamples lower resolution feature map to match higher resolution map.

```text
THOR  36 x 36  - unchanged 
DINO 18 x 18 - upsampled to 36 x 36
```

`backbone_rescale_features: down` downsamples the higer resolution feature map to math the lower resolution feature map.

```text
THOR  36 x 36  - downsampled to 18 x 18
DINO 18 x 18 - unchanged
```
