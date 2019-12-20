import torch
import torch.nn as nn
from networks.default import Generator, Discriminator, Siamese
from losses import AdversarialLoss, TravelLoss, MarginContrastiveLoss
from losses import compute_gp
from torch.optim import Adam
from utils import initialize_weights
import os


class TravelGAN(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf, dropout,
                 ndf, n_layers_dis, nsf, n_layers_siam, latent_dim,
                 lr, margin, lambda_adv, lambda_travel, lambda_margin,
                 lambda_gp, type, sn, device="cpu"):
        super(TravelGAN, self).__init__()
        # Parameters
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.ngf = ngf
        self.dropout = dropout
        self.ndf = ndf
        self.n_layers_dis = n_layers_dis
        self.nsf = nsf
        self.n_layers_siam = n_layers_siam
        self.laten_dim = latent_dim
        self.lr = lr
        self.margin = margin
        self.device = device
        self.lambda_adv = lambda_adv
        self.lambda_travel = lambda_travel
        self.lambda_margin = lambda_margin
        self.lambda_gp = lambda_gp
        self.type = type
        self.sn = sn

        # Modules
        self.gen_ab = Generator(input_nc, output_nc,
                                num_downs, ngf, dropout, sn)
        self.gen_ba = Generator(input_nc, output_nc,
                                num_downs, ngf, dropout, sn)
        self.dis_a = Discriminator(input_nc, ndf, n_layers_dis, sn)
        self.dis_b = Discriminator(input_nc, ndf, n_layers_dis, sn)
        # self.siam_a = Siamese(input_nc, nsf, n_layers_siam, latent_dim)
        # self.siam_b = Siamese(input_nc, nsf, n_layers_siam, latent_dim)
        self.siam = Siamese(input_nc, nsf, n_layers_siam, latent_dim)

        # Optimizers
        dis_params = list(self.dis_a.parameters()) + \
            list(self.dis_b.parameters())
        gen_params = list(self.gen_ab.parameters()) + \
            list(self.gen_ba.parameters()) + list(self.siam.parameters())
            # list(self.siam_a.parameters()) + \
            # list(self.siam_b.parameters())
        self.dis_optim = Adam([p for p in dis_params], lr=lr, betas=(0.5, 0.9))
        self.gen_optim = Adam([p for p in gen_params], lr=lr, betas=(0.5, 0.9))

        # Losses
        self.adv_loss = AdversarialLoss(type, device)
        if type == "wgangp":
            self.gp = compute_gp
        self.travel_loss = TravelLoss()
        self.margin_loss = MarginContrastiveLoss(margin)

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
        adv_loss = self.adv_loss(self.dis_a(x_a), True) + \
            self.adv_loss(self.dis_b(x_b), True) + \
            self.adv_loss(self.dis_b(x_ab), False) + \
            self.adv_loss(self.dis_a(x_ba), False)
        dis_loss = self.lambda_adv * adv_loss
        if self.type == "wgangp":
            gp = self.gp(self.dis_a, x_a, x_ba) + \
                self.gp(self.dis_b, x_b, x_ab)
            dis_loss += self.lambda_gp * gp
        dis_loss.backward()
        self.dis_optim.step()
        return dis_loss.item()

    # def gen_update(self, x_a, x_b):
    #     self.gen_optim.zero_grad()
    #     x_ab = self.gen_ab(x_a)
    #     x_ba = self.gen_ba(x_b)
    #     adv_loss = self.adv_loss(self.dis_b(x_ab), True) + \
    #         self.adv_loss(self.dis_a(x_ba), True)
    #     travel_loss = self.travel_loss(x_a, x_ba, self.siam_a) + \
    #         self.travel_loss(x_b, x_ab, self.siam_b)
    #     margin_loss = self.margin_loss(
    #         x_a, self.siam_a) + self.margin_loss(x_b, self.siam_b)
    #     gen_loss = self.lambda_adv * adv_loss + \
    #         self.lambda_travel * travel_loss + \
    #         self.lambda_margin * margin_loss
    #     gen_loss.backward()
    #     self.gen_optim.step()
    #     return gen_loss.item()

    def gen_update(self, x_a, x_b):
        self.gen_optim.zero_grad()
        x_ab = self.gen_ab(x_a)
        x_ba = self.gen_ba(x_b)
        adv_loss = self.adv_loss(self.dis_b(x_ab), True) + \
            self.adv_loss(self.dis_a(x_ba), True)
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
