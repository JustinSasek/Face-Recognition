# import torch.nn as nn
# import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(
            3, 3), padding='same')   # only need inside constructor - no padding by default
        self.relU1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 112
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(3, 3), padding='same')
        self.relU2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 56
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=24, kernel_size=(3, 3), padding='same')
        self.relU3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 28
        self.conv4 = nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=(3, 3), padding='same')
        self.relU4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 14
        self.conv5 = nn.Conv2d(
            in_channels=48, out_channels=80, kernel_size=(3, 3), padding='same')
        self.relU5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.pad = nn.ReplicationPad2d((0, 1, 0, 1))
        # 8
        self.conv6 = nn.Conv2d(
            in_channels=80, out_channels=100, kernel_size=(3, 3), padding='same')
        self.relU6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 4
        self.conv7 = nn.Conv2d(
            in_channels=100, out_channels=100, kernel_size=(3, 3), padding='same')
        self.relU7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        # 1
        self.flat1 = nn.Flatten()
        self.dense = nn.Linear(in_features=100, out_features=1)
        self.sigmoid = nn.Sigmoid()
        # self.flat2 = nn.Flatten(start_dim=0)

        # self.softMax = nn.Softmax()

    def forward(self, x):  # x (N,C,H,W)
        x = self.conv1(x)
        x = self.relU1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relU2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relU3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relU4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relU5(x)
        x = self.pool5(x)
        x = self.pad(x)
        x = self.conv6(x)
        x = self.relU6(x)
        x = self.pool6(x)
        x = self.conv7(x)
        x = self.relU7(x)
        x = self.pool7(x)
        x = self.flat1(x)
        x = self.dense(x)
        # x = self.flat2(x)
        # x = self.softMax(x)
        return x


# def model():
#     net = None

#     return net


# model()
