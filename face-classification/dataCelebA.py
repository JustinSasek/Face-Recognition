import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

target_id = 667  # identity we will be looking for
batch_size = 128  # batch size the dataloader will use

img_transform = T.Compose([T.Resize(256),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ])  # transformations done on the dataset during pretraining on imagenet


def identify(identity):
    return np.float32(1) if identity == target_id else np.float32(0)


datasets = {x: datasets.CelebA(".",
                               split=x,
                               target_type="identity",
                               transform=img_transform,
                               target_transform=identify
                               )
            for x in ['train', 'valid']
            }
dataloaders = {x: DataLoader(datasets[x],
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4
                             )
               for x in ['train', 'valid']
               }
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'valid']}


def data():
    return dataloaders, dataset_sizes
