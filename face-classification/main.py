from faceDataloader import data
from model import FaceClassifierCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

dataLoaders, dataset_sizes = data(256) #946 images total


def train():
    training_data = dataLoaders['train']
    num_epochs = 4;

    for epoch in range(num_epochs):
        ### TRAINING PHASE ###
        print(f'Epoch{epoch} / {num_epochs - 1}')

        for inputs, labels in training_data:

            pass

        ### VALIDATION PHASE ###

        for inputs, labels in training_data:
            pass


if __name__ == '__main__':
    # print('Hello World!')
    # print(dataLoaders)
    print(dataset_sizes)

    training_data = dataLoaders['train']
    validation_data = dataLoaders['val']
    testing_data = dataLoaders['test']
    # print(training_data)

    # for labels in training_data:
        # print(labels)
    # for inputs, labels in validation_data:
    #     # print(len(inputs), len(labels))
    #     print(inputs.shape, labels.shape)
    # for inputs, labels in testing_data:
    #     # print(len(inputs), len(labels))
    #     print(inputs.shape, labels.shape)
    
    net = FaceClassifierCNN()
    for inputs, labels in training_data:
        loss = nn.L1Loss()
        input = net.forward(inputs)
        target = labels
        output = loss(input, target)
        print("input")
        print(input)
        print("target")
        print(target)
        output.backward()
