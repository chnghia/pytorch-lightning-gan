import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pytorch_lightning.trainer import Trainer
from models.cyclegan_model import CycleGanModel


def main(img_height: int = 256, img_width: int = 256, dataset_name="mini", batch_size: int = 1, n_cpu: int = 4) -> None:

    model = CycleGanModel()
    trainer = Trainer(gpus=[0], precision=16, amp_level="O1", max_epochs=100)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    hparams = parser.parse_args()
    main(hparams)
