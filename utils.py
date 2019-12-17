import torch
import torch.nn as nn
import os
import json
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def spectral_normalization(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m = nn.utils.spectral_norm(m)


def nparams(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad_])


def get_device(device):
    device_idx = device
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(device_idx))
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    print('Launching experiment on device : {}'.format(device))
    return device


def get_writer(fdir):
    print("Loading writer..")
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    return SummaryWriter(fdir), 'Epoch {} | Gen : {} | Dis : {}'


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, "{}.json".format(name)), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, "{}.json".format(name))
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def visualize_batch(x):
    x = make_grid(x.detach().cpu(), normalize=True, range=(-1, 2))
    x = x.numpy().transpose((1, 2, 0))
    plt.imshow(x)
