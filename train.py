import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from models.pix2pix_model import Pix2PixModel
from pytorch_lightning.trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from models.pix2pix.datasets import ImageDataset


def main(img_height: int = 256,
         img_width: int = 256,
         dataset_name="mini_pix2pix",
         batch_size: int = 1,
         n_cpu: int = 4) -> None:

    model = Pix2PixModel()

    trainer = Trainer(gpus=[0])

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    hparams = parser.parse_args()
    main(hparams)
