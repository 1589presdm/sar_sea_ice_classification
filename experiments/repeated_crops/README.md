# Repeated Crops

This experiment increases the number of random training crops sampled from each image during one epoch.

## Files 

- `repeated_dataset.py` small dataset wrappers that repeats the original dataset
- `repeated_datamodule.py` defines `RepeatedCropMiltiModalDataModule`, which inherits from `GenericMultiModalDataModule` and wraps only the training dataset

## Training

Before starting training update the dataset paths.

Run training:

```bash
PYTHONPATH=. uv run terratorch fit \
    --config experiments/repeated_crops/S1_multibackbone_multimodal_repeated.yaml
```
PYTHONPATH=. required that TerraTorch can import local module

## Testing

Run testing:

```bash
PYTHONPATH=. uv run terratorch test \
    --config experiments/repeated_crops/S1_multibackbone_multimodal_repeated.yaml \
    --ckpt_path /path/to/checkoint001.ckpt
```

## Notebook experiment

The notebook experiment was run with the next configuration:

```text
MAX_EPOCH = 200
BATCH_SIZE = 4
TRAIN_REPEATS = 3
```

Original train dataset contained 136 images, after applying repeated crops training dataset contained 408 samples per epoch.

Traing lasted 11 hours and 28 miutes and stopped after 157 epochs (logget value 16.01k training batches).

The final metrics and prediction visualization available in .ipynb files