from data import get_datasets
from trainer import TravelGAN
from torch.utils.data.dataloader import DataLoader
from utils import get_device, load_json, get_writer
import argparse
from statistics import mean


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=int, help="gpu id")
parser.add_argument("-n", "--log", type=str, help="name of log folder")
parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
opts = parser.parse_args()


# Get CUDA/CPU device
device = get_device(opts.device)

print('Loading data..')
hparams = load_json('./configs', opts.hparams)
dataset_a, dataset_b = get_datasets(**hparams['dataset'])
loader_a = DataLoader(dataset_a, **hparams['loading'])
loader_b = DataLoader(dataset_b, **hparams['loading'])
model = TravelGAN(**hparams['model'], device=device)
writer, monitor = get_writer(opts.log)

print('Start training..')
for epoch in range(hparams['n_epochs']):
    # Run one epoch
    dis_losses, gen_losses = [], []
    for x_a, x_b in zip(loader_a, loader_b):
        # Loading on device
        x_a = x_a.to(device, non_blocking=True)
        x_b = x_b.to(device, non_blocking=True)

        # Calculate losses and update weights
        dis_loss = model.dis_update(x_a, x_b)
        gen_loss = model.gen_update(x_a, x_b)
        dis_losses.append(dis_loss)
        gen_losses.append(gen_loss)

    # Logging losses
    dis_loss, gen_loss = mean(dis_losses), mean(gen_losses)
    writer.add_scalar('dis', dis_loss, epoch)
    writer.add_scalar('gen', gen_loss, epoch)
    print(monitor.format(epoch, gen_loss, dis_loss))

    # Saving model every n_save_steps epochs
    if (epoch + 1) % hparams['n_save_steps'] == 0:
        model.save(opts.log, epoch)
