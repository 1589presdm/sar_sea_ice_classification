from torch.utils.data import Dataset


class RepeatedDataset(Dataset):
    """
    Repeat existing dataset multiple times per epoch.
    """

    def __init__(self, dataset: Dataset, repeats: int = 1):
        self.dataset = dataset # original size of the dataset 
        self.repeats = repeats # number of times to repeat dataset per epoch

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, index):
        real_index = index % len(self.dataset) # cycle through original dataset indices 
        return self.dataset[real_index] # calls originall dataset again