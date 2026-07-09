# SAR sea ice and water classification 

Multimodal semantic segmentation of sea ice and water using Sentinel-1 SAR imagery and Doppler channel.

The oroject uses:

- THOR ViT for Sentinel-1 HH/HV channels;
- DINO ViT for Doppler channel;

The model predicts two classes:

```text
0 = ice
1 = water
```

Pixels labeled `-1` are ignored during loss and metric calculation.

The main derictories are:

- `configs/` contains the TerraTorch configuration and CLI instructions;
- `notebooks/` contains training and evaluation examples;
- `data_preparation/` contains data preparation scripts;
- `thor_extension/` proivede registrartion of the installed THOR TerraTorch extension.


## Installation

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

Clone the repository and install all dependencies:

```bash
uv sync
```

The command creates local environment and install dependecies defind in `pyproject.toml`.

## Dataset

The project expects the dataset structure:

```text
DATA_ROOT/
    tif_images/
    tif_doppler/
    tif_mask/
    splits/
        train.txt
        val.txt
        test.txt
```

Input modalities:

```text
Sentinel-1:
    EW_HH
    EW_HV

Doppler:
    Doppler channel
```