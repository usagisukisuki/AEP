import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import random
from PIL import Image

class RPECellDataloader(data.Dataset):
    def __init__(self, root=None, cross=None, dataset_type='train',transform=None):
        self.root = root
        self.dataset_type = dataset_type
        self.cross = cross
        self.transform = transform

        self.data = np.load(self.root + "/{}_images_{}.npy".format(self.dataset_type, self.cross))
        self.label = np.load(self.root + "/{}_labels_{}.npy".format(self.dataset_type, self.cross))


    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        label = np.where(label[0]==255, 1, 0)
        image = Image.fromarray(np.uint8(image[0]*255.0)).convert("L")
        label = Image.fromarray(np.uint8(label)).convert("L")

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self):
        return len(self.data)



