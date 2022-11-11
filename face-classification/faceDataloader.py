import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.io import read_image
import pandas as pd
import os
import random


target_id = 1  # identity we will be looking for
batch_size = 2  # batch size the dataloader will use


class FaceDataset(Dataset):
    """Face Dataset."""
    TRAIN_VAL_TEST_SPLIT = (0.65, 0.2, 0.15)  # 65% training, 20% validation, 15% testing

    def __init__(self, annotations_file, img_dir, split='all', size_file=None, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.img_labels = pd.read_csv(annotations_file, delim_whitespace=True, names=('img_file', 'id'))

        if self.split != 'all':
            if size_file is None:  # get size of dataset, either from file or calculated from labels
                self.n = sum(1 for line in open(annotations_file))
            else:
                with open(size_file, 'r') as f:
                    self.n = int(f.read().strip())
            self.sets = {'all': set(i for i in range(self.n))}  # our entire dataset

            random.seed(1)

            self.sets['test'] = self.sets['all'].copy()  # by the end this will be only the test set
            self.sets['train'] = set(random.sample(self.sets['test'], int(self.n * self.TRAIN_VAL_TEST_SPLIT[0])))
            self.sets['test'] -= self.sets['train']  # remove training samples from test set
            self.sets['val'] = set(random.sample(self.sets['test'], int(self.n * self.TRAIN_VAL_TEST_SPLIT[1])))
            self.sets['test'] -= self.sets['val']  # remove validation samples from dataset, leaving only the test set

            self.img_labels = self.img_labels.iloc[list(self.sets[self.split])]  # select only the corresponding set

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


def data(batch_size = 256):
    # Transformations done on the dataset during pretraining on imagenet
    img_transform = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

    root_dir = './eclair-faces/'
    img_dir = os.path.join(root_dir, "img")
    id_file = os.path.join(root_dir, "id.txt")
    size_file = os.path.join(root_dir, "size.txt")

    datasets = {x: FaceDataset(id_file,
                               img_dir,
                               split=x,
                               size_file=size_file,
                               transform=img_transform,
                               target_transform=np.float32,  # convert longs to float32
                               )
                for x in ['train', 'val', 'test']
                }
    dataloaders = {x: DataLoader(datasets[x],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4
                                 )
                   for x in ['train', 'val', 'test']
                   }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes
