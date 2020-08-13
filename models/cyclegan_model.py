import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import numpy as np
from PIL import Image
import itertools

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

from models.cyclegan.models import GeneratorResNet, Discriminator
from models.cyclegan.utils import ReplayBuffer, LambdaLR
from models.cyclegan.datasets import ImageDataset


cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.HalfTensor if cuda else torch.FloatTensor


class CycleGanModel(LightningModule):
    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 2,
                 img_height: int = 128,
                 img_width: int = 128,
                 channels: int = 3,
                 lambda_pixel: int = 100,
                 n_cpu: int = 4,
                 n_residual_blocks: int = 7,
                 lambda_cyc: float = 10.0,
                 lambda_id: float = 5.0,
                 dataset_name="horse2zebra", **kwargs):
        super().__init__()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.dataset_name = dataset_name
        self.n_cpu = n_cpu
        input_shape = (channels, img_height, img_width)
        self.input_shape = input_shape
        self.n_residual_blocks = n_residual_blocks
        self.lambda_cyc = lambda_cyc
        self.lambda_id = lambda_id

        # Image transformations
        self.transforms_ = [
            transforms.Resize(int(self.img_height * 1.12), Image.BICUBIC),
            transforms.RandomCrop((self.img_height, self.img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.G_AB = GeneratorResNet(input_shape, self.n_residual_blocks)
        self.G_BA = GeneratorResNet(input_shape, self.n_residual_blocks)
        self.D_A = Discriminator(input_shape)
        self.D_B = Discriminator(input_shape)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def forward(self, z):
        return self.G_AB(z)

    def adversarial_loss(self, y_hat, y):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(
            Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
        fake = Variable(
            Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # ------------------
        #  Train Generators
        # ------------------
        if optimizer_idx == 0:

            # Arange images along x-axis
            real_A_ = torchvision.utils.make_grid(
                real_A, nrow=5, normalize=True)
            real_B_ = torchvision.utils.make_grid(
                real_B, nrow=5, normalize=True)
            fake_A_ = torchvision.utils.make_grid(
                fake_A, nrow=5, normalize=True)
            fake_B_ = torchvision.utils.make_grid(
                fake_B, nrow=5, normalize=True)
            # Arange images along y-axis
            image_grid = torch.cat((real_A_, fake_B_, real_B_, fake_A_), 1)
            grid = torchvision.utils.make_grid(image_grid)
            self.logger.experiment.add_image('generated_images', grid, 0)

            self.G_AB.train()
            self.G_BA.train()

            # Identity loss
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = self.G_BA(fake_B)
            loss_cycle_A = self.criterion_cycle(recov_A, real_A)
            recov_B = self.G_AB(fake_A)
            loss_cycle_B = self.criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
            tqdm_dict = {'loss_G': loss_G}

            output = OrderedDict({
                'loss': loss_G,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx > 0:
            # Real loss
            loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = self.criterion_GAN(
                self.D_A(fake_A_.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

            # -----------------------
            #  Train Discriminator A
            # -----------------------
            if optimizer_idx == 1:
                # Real loss
                # loss_real = self.criterion_GAN(self.D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                # fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
                # loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
                # Total loss
                # loss_D_A = (loss_real + loss_fake) / 2

                tqdm_dict = {'loss_D_A': loss_D_A}

                output = OrderedDict({
                    'loss': loss_D_A,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                return output

            # -----------------------
            #  Train Discriminator B
            # -----------------------
            if optimizer_idx == 2:
                # Real loss
                loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
                loss_fake_B = self.criterion_GAN(
                    self.D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real_B + loss_fake_B) / 2

                loss_D = (loss_D_A + loss_D_B) / 2

                tqdm_dict = {'loss_D': loss_D}

                output = OrderedDict({
                    'loss': loss_D,
                    'progress_bar': tqdm_dict,
                    'log': tqdm_dict
                })
                return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        # Optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(b1, b2)
        )
        optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=lr, betas=(b1, b2))

        # Learning rate update schedulers
        #lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_G, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)
        #lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_D_A, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)
        #lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        #    optimizer_D_B, lr_lambda=LambdaLR(
        #        self.n_epochs, self.epoch, self.decay_epoch).step
        #)

        return [optimizer_G, optimizer_D_A, optimizer_D_B], []

    def train_dataloader(self):
        # Training data loader
        dataloader = DataLoader(
            ImageDataset("./data/%s" % self.dataset_name,
                         transforms_=self.transforms_, unaligned=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_cpu,
        )
        return dataloader

#     def val_dataloader(self):
#         val_dataloader = DataLoader(
#             ImageDataset("./data/%s" % self.dataset_name,
#                          transforms_=self.transforms_, unaligned=True, mode="test"),
#             batch_size=1,
#             shuffle=True,
#             num_workers=1,
#         )
#         return val_dataloader

    def on_epoch_end(self):
        pass

#     def validation_step(self, batch, batch_idx):
#         pass

#     def validation_epoch_end(self, outputs):
#         pass
