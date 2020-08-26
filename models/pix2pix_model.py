import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from models.pix2pix.models import GeneratorUNet, Discriminator
from PIL import Image
from models.pix2pix.datasets import ImageDataset

import albumentations as A

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Pix2PixModel(LightningModule):
    def __init__(
        self,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 1,
        img_height: int = 256,
        img_width: int = 256,
        lambda_pixel: int = 100,
        n_cpu: int = 4,
        dataset_name="mini_pix2pix",
        **kwargs
    ):
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
        self.dataset_name = dataset_name
        self.n_cpu = n_cpu

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
        # Configure dataloaders
#         transforms_ = [
#             transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
        self.transforms_ = [
            A.RandomResizedCrop(
                self.img_height,
                self.img_width,
            ),
            A.Rotate(13),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma(),
            # A.CLAHE(),
#             MyCoarseDropout(
#                 min_holes=1,
#                 max_holes=8,
#                 max_height=32,
#                 max_width=32,
#             ),
            # A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
#             A.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225],
#             ),
            # ToTensorV2(),
        ]

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        raise NotImplementedError

    def set_input(self, input):
        self.real_A = input["A"]
        self.real_B = input["B"]
        # self.image_paths = input["A_paths"]

        # Model inputs
#         self.real_A = Variable(input["B"].type(Tensor))
#         self.real_B = Variable(input["A"].type(Tensor))

        # Adversarial ground truths
        self.valid = Variable(Tensor(np.ones((self.real_A.size(0), *self.patch))), requires_grad=False)
        self.fake = Variable(Tensor(np.zeros((self.real_A.size(0), *self.patch))), requires_grad=False)

    def update_loss_D(self):
        raise NotImplementedError

    def update_load_G(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Model inputs
        self.set_input(batch)
        print(self.real_A.shape)

        # generate images
        self.fake_B = self.forward(self.real_A)

        # train generator
        if optimizer_idx == 0:
            # log sampled images
            img_sample = torch.cat((self.real_A, self.fake_B, self.real_B), -2)
            grid = torchvision.utils.make_grid(img_sample)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # GAN loss
            # fake_B = self.generator(real_A)
            pred_fake = self.discriminator(self.fake_B, self.real_A)
            loss_GAN = self.criterion_GAN(pred_fake, self.valid)
            # Pixel-wise loss
            loss_pixel = self.criterion_pixelwise(self.fake_B, self.real_B)

            # Total loss
            g_loss = loss_GAN + self.lambda_pixel * loss_pixel
            tqdm_dict = {"g_loss": g_loss}

            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # self.optimizer_D.zero_grad()

            # Real loss
            pred_real = self.discriminator(self.real_B, self.real_A)
            loss_real = self.criterion_GAN(pred_real, self.valid)

            # Fake loss
            pred_fake = self.discriminator(self.fake_B.detach(), self.real_A)
            loss_fake = self.criterion_GAN(pred_fake, self.fake)

            # Total loss
            d_loss = 0.5 * (loss_real + loss_fake)
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        # Optimizers
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Schedulers
        gen_sched = {'scheduler': lr_scheduler.ExponentialLR(opt_g, 0.99),
                 'interval': 'step'}  # called after each training step
        dis_sched = lr_scheduler.CosineAnnealingLR(opt_d, T_max=10) # called every epoch

        return [opt_g, opt_d], [gen_sched, dis_sched]

    def train_dataloader(self):
        dataloader = DataLoader(ImageDataset("./datasets/%s" % self.dataset_name, transforms_=self.transforms_), 
                                batch_size=self.batch_size, shuffle=True, num_workers=self.n_cpu,)
        return dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(ImageDataset("./datasets/%s" % self.dataset_name, transforms_=self.transforms_, mode="val"), 
                                    batch_size=10, shuffle=True, num_workers=1,)
        return val_dataloader

    def on_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass
