import torch
from torch.utils.data import Dataset
from torch.distributions import Distribution


class GeneratorDataset(Dataset):
    def __init__(self, function, input_dist: Distribution, size: int):
        self.distribution = input_dist
        self.size = size
        self.samples = self.distribution.sample((size,))

        self.samples = self.samples.view(size, 1, -1)
        self.values = [function(*sample[0]) for sample in self.samples]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx], self.values[idx]
