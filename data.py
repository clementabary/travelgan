from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np


class CIFARSubset(Dataset):
    def __init__(self, data_path, label):
        dataset = CIFAR10(data_path)
        targets = np.array(dataset.targets)
        indices = np.where(targets == dataset.class_to_idx[label])
        self.data = [dataset.data[indices] for i in indices][0]
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)


def get_datasets(data_path, label_a, label_b):
    dataset_a = CIFARSubset(data_path, label_a)
    dataset_b = CIFARSubset(data_path, label_b)
    return dataset_a, dataset_b
