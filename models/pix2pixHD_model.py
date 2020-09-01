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
from PIL import Image

import models.networks.pix2pxhd_nets as networks

import albumentations as A

cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class BaseOptions:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Pix2PixHDModel(LightningModule):
    def __init__(self, opt):
        super().__init__()

        netG_input_nc, use_sigmoid, netD_input_nc = self.set_params(opt)

        self.netG = networks.define_G(
            netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids
        )
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, "encoder", opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

    def set_params(self, opt):
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num

        use_sigmoid = opt.no_lsgan
        netD_input_nc = input_nc + opt.output_nc
        if not opt.no_instance:
            netD_input_nc += 1

        return netG_input_nc, use_sigmoid, netD_input_nc

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        raise NotImplementedError

    def set_input(self, input):
        raise NotImplementedError

    def update_loss_D(self):
        raise NotImplementedError

    def update_load_G(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass
