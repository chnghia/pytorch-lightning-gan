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
         n_cpu: int = 8) -> None:

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset("./data/%s" %
                     dataset_name, transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    val_dataloader = DataLoader(
        ImageDataset("./data/%s" % dataset_name,
                     transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=True,
        num_workers=1,
    )

    model = Pix2PixModel()

    trainer = Trainer(gpus=[0])

    trainer.fit(model, dataloader, val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
