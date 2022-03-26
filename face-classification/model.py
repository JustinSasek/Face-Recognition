import torch.nn as nn
import torchvision.models as models


def model():
    net = models.efficientnet_b0(pretrained=True)

    for param in net.parameters():
        param.requires_grad = False
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=1, bias=True),
        nn.Sigmoid()
    )

    return net


model()
