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