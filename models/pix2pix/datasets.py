import glob
import random
import os
import numpy as np
import re
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = A.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)
#         return {"A": img_A, "B": img_B}

        img_A = np.array(img_A)
        img_B = np.array(img_B)

        augmented = self.transform(image=img_A, mask=img_B)
        img = np.array(augmented['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(augmented['mask']).astype(np.float32).transpose((2, 0, 1))

        return {"A": img / 255., "B": mask / 255.}

    def __len__(self):
        return len(self.files)

class FloorplanDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = A.Compose(transforms_)
        files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        regex = re.compile(r'\d+.jpg')
        self.files = list(filter(regex.search, files))

    def get_item(self, path):
        filename, file_extension = os.path.splitext(path)
        return path, "{}_multi.png".format(filename)

    def __getitem__(self, index):
        path_A, path_B = self.get_item(self.files[index % len(self.files)])
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        img_A = np.array(img_A)
        img_B = np.array(img_B)
#         print(img_A.shape, img_B.shape)
        augmented = self.transform(image=img_A, mask=img_B)
        img = np.array(augmented['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(augmented['mask']).astype(np.float32).transpose((2, 0, 1))

        return {"A": img / 255., "B": mask / 255.}

    def __len__(self):
        return len(self.files)