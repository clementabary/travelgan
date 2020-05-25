import torch
import torch.nn as nn
from networks.default import Generator, Discriminator, Siamese
from losses import AdversarialLoss, TravelLoss, MarginContrastiveLoss
from losses import compute_gp
from torch.optim import Adam
from utils import initialize_weights
import os


class TravelGAN(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super(TravelGAN, self).__init__()
        # Parameters
        self.hparams = hparams
        self.device = device

        # Modules
        self.gen_ab = Generator(**hparams["gen"])
        self.gen_ba = Generator(**hparams["gen"])
        self.dis_a = Discriminator(**hparams["dis"])
        self.dis_b = Discriminator(**hparams["dis"])
        self.siam = Siamese(**hparams["siam"])

        # Loss coefficients
        self.lambda_adv = hparams["lambda_adv"]
        self.lambda_travel = hparams["lambda_travel"]
        self.lambda_margin = hparams["lambda_margin"]
        self.margin = hparams["margin"]
        self.lambda_gp = hparams["lambda_gp"]
        self.type = hparams["type"]

        # Learning rates
        self.lr_dis = hparams["lr_dis"]
        self.lr_gen = hparams["lr_gen"]

        # Optimizers
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + \
            list(self.gen_ba.parameters()) + list(self.siam.parameters())
        self.dis_optim = Adam([p for p in dis_params],
                              lr=self.lr_dis, betas=(0.5, 0.999))
        self.gen_optim = Adam([p for p in gen_params],
                              lr=self.lr_gen, betas=(0.5, 0.999))

        # Losses
        self.adv_loss = AdversarialLoss(self.type, device)
        if self.type == "wgangp":
            self.gp = compute_gp
        self.travel_loss = TravelLoss()
        self.margin_loss = MarginContrastiveLoss(self.margin)

        # Initialization
        self.apply(initialize_weights)
        self.set_to(device)

    def forward(self, x_a, x_b):
        self.eval()
        return self.gen_ab(x_a), self.gen_ba(x_b)

    def dis_update(self, x_a, x_b):
        self.dis_optim.zero_grad()
        x_ab = self.gen_ab(x_a).detach()
        x_ba = self.gen_ba(x_b).detach()
        adv_loss = self.adv_loss(self.dis_a(x_a), True, True) + \
            self.adv_loss(self.dis_b(x_b), True, True) + \
            self.adv_loss(self.dis_b(x_ab), False, True) + \
            self.adv_loss(self.dis_a(x_ba), False, True)
        dis_loss = self.lambda_adv * 0.5 * adv_loss
        if self.type == "wgangp":
            gp = self.gp(self.dis_a, x_a, x_ba) + \
                self.gp(self.dis_b, x_b, x_ab)
            dis_loss += self.lambda_gp * gp
        dis_loss.backward()
        self.dis_optim.step()
        return dis_loss.item()

    def gen_update(self, x_a, x_b):
        self.gen_optim.zero_grad()
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b)
        adv_loss = self.adv_loss(self.dis_b(x_ab), True, False) + \
            self.adv_loss(self.dis_a(x_ba), True, False)
        travel_loss = self.travel_loss(x_a, x_ab, self.siam) + \
            self.travel_loss(x_b, x_ba, self.siam)
        margin_loss = self.margin_loss(
            x_a, self.siam) + self.margin_loss(x_b, self.siam)
        gen_loss = self.lambda_adv * adv_loss + \
            self.lambda_travel * travel_loss + \
            self.lambda_margin * margin_loss
        gen_loss.backward()
        self.gen_optim.step()
        return gen_loss.item()

    def resume(self, file):
        state_dict = torch.load(file, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, checkpoint_dir, epoch):
        file = 'model_{}.pt'.format(epoch + 1)
        file = os.path.join(checkpoint_dir, file)
        torch.save(self.state_dict(), file)

    def set_to(self, device):
        self.device = device
        self.to(device)
        print("Model loaded on device : {}".format(device))
