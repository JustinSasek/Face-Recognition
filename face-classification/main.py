import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import model as m
from faceDataloader import data
import time
import copy
import os


MODEL_NAME = 'model'

dataloaders, dataset_sizes = data()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    return model


if __name__ == '__main__':
    print("Training")
    # train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)



