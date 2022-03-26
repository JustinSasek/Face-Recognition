import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T
from torchvision.io import read_image
import pandas as pd
import os
import torch

target_id = 1  # identity we will be looking for

class FaceDataset(Dataset):
    """Face Dataset."""

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, delim_whitespace=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Return the number of images.
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Return one image and its target id.
        """
        # Read in image and label
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        # Apply transformations on image and id
        if self.transform:
            image = self.transform(image.float())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def identify(identity):
    """
    Checks if an id matches the id for images with a face
    """
    return np.float32(1) if identity == target_id else np.float32(0)

if __name__ == "__main__":
    batch_size = 2  # batch size the dataloader will use
    root_dir = './eclair-faces'
    img_dir = os.path.join(root_dir, "img")
    id_file = os.path.join(root_dir, "id.txt")
    size_file = os.path.join(root_dir, "size.txt")

    data = FaceDataset(id_file, img_dir)