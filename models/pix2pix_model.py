import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from models.pix2pix.models import GeneratorUNet, Discriminator

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Pix2PixModel(LightningModule):
    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 img_height: int = 256,
                 img_width: int = 256,
                 lambda_pixel: int = 100, ** kwargs):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()

        # Loss weight of L1 pixel-wise loss between translated image and real image
        self.lambda_pixel = lambda_pixel

        # Loss functions
        # self.criterion_GAN = torch.nn.MSELoss()
        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        # imgs, _ = batch

        # Model inputs
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)
        
        # generate images
        fake_B = self.generator(real_A)

        # train generator
        if optimizer_idx == 0:
            # log sampled images
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
            grid = torchvision.utils.make_grid(img_sample)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # GAN loss
            # fake_B = self.generator(real_A)
            pred_fake = self.discriminator(fake_B, real_A)
            loss_GAN = self.criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(fake_B, real_B)

            # Total loss
            g_loss = loss_GAN + self.lambda_pixel * loss_pixel
            tqdm_dict = {'g_loss': g_loss}

            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # self.optimizer_D.zero_grad()

            # Real loss
            pred_real = self.discriminator(real_B, real_A)
            loss_real = self.criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = self.discriminator(fake_B.detach(), real_A)
            loss_fake = self.criterion_GAN(pred_fake, fake)

            # Total loss
            d_loss = 0.5 * (loss_real + loss_fake)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        # Optimizers
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        pass
        
    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass
