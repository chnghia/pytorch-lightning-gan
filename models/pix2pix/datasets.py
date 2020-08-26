import glob
import random
import os
import numpy as np

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
            
        img_A = np.array(img_A)
        img_B = np.array(img_B)

        augmented = self.transform(image=img_A, mask=img_B)
#         print(augmented)
#         img_A = self.transform(augmented['image'])
#         img_B = self.transform(augmented['mask'])
        img = np.array(augmented['image']).astype(np.float32) #.transpose((2, 0, 1))
        mask = np.array(augmented['mask']).astype(np.float32)

        return {"A": img / 255., "B": mask / 255.}

    def __len__(self):
        return len(self.files)
