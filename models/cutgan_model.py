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


class CUTGanModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
