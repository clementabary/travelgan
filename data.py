from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Resize, ToTensor, Normalize
from torchvision.transforms import Compose
from PIL import Image
import numpy as np
import glob


class CIFARSubset(Dataset):
    def __init__(self, data_path, label, n_tracks):
        dataset = CIFAR10(data_path)
        targets = np.array(dataset.targets)
        indices = np.where(targets == dataset.class_to_idx[label])
        self.data = [dataset.data[indices] for i in indices][0]
        self.data = self.data[:n_tracks]
        self.transform = get_transform(False, True, True, False)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)


class ImageNetSubset(Dataset):
    def __init__(self, data_path, label, n_tracks):
        dataset = sorted(glob.glob(data_path + '/{}/*/*.jpg'.format(label)))
        self.data = []
        for idx, file in enumerate(dataset):
            im = Image.open(file).convert('RGB')
            self.data.append(im.copy())
            im.close()
            if idx + 1 >= n_tracks:
                break
        self.transform = get_transform(True, True, True, True)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])

    def __len__(self):
        return len(self.data)


def get_datasets(type, data_path, label_a, label_b, n_tracks=5000):
    if type == "cifar":
        dataset_a = CIFARSubset(data_path, label_a, n_tracks)
        dataset_b = CIFARSubset(data_path, label_b, n_tracks)
    elif type == "imagenet":
        dataset_a = ImageNetSubset(data_path, label_a, n_tracks)
        dataset_b = ImageNetSubset(data_path, label_b, n_tracks)
    return dataset_a, dataset_b


def get_transform(resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(CenterCrop(256))
        if resize:
            options.append(Resize((128, 128)))
        if totensor:
            options.append(ToTensor())
        if normalize:
            options.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = Compose(options)
        return transform
