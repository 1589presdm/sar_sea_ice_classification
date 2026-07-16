from terratorch.datamodules import GenericMultiModalDataModule

from experiments.repeated_crops.repeated_dataset import RepeatedDataset


class RepeatedCropMultiModalDataModule(GenericMultiModalDataModule):
    """
    GenericMultiModalDataModule with repeated training samples.
    This increases the number of random crops sampled from each training image
    during one epoch.
    """

    def __init__(self, train_repeats: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.train_repeats = train_repeats
    
    # prepare train/val/test dataset using parent TerraTorch DataModule
    def setup(self, stage=None):
        super().setup(stage) 

        # wrap only the training ddataset, only once and when repeat is needed 
        if (
            hasattr(self, "train_dataset")
            and self.train_dataset is not None
            and self.train_repeats > 1
            and not isinstance(self.train_dataset, RepeatedDataset)
        ):
            original_length = len(self.train_dataset) # train_datasetis created in the parent setup() method

            self.train_dataset = RepeatedDataset(
                dataset=self.train_dataset,
                repeats=self.train_repeats,
            )

            print(
                f"repeated train dataset: "
                f"{original_length} images x {self.train_repeats} repeats = {len(self.train_dataset)} samples per epoch"
            )