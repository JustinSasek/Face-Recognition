import torch
import torchvision.models as models


def model():
    net = models.efficientnet_b0(pretrained=True)

    return net


model()
