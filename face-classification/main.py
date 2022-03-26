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
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_1 = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.view(-1, 1).to(device)
                num_1 += torch.count_nonzero(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # preds = outputs.clone().to(device)  # so that when we round we dont round the actual outputs which need grad
                    preds = torch.round(outputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc} 1: {num_1} 0: {dataset_sizes[phase] - num_1}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train():
    model = m()
    model.to(device)

    criterion = nn.BCELoss()  # loss

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), f'{MODEL_NAME}.pt'))

    return model


def load():
    model = m()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), f'{MODEL_NAME}.pt')))

    return model


def test(model):
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    num_1 = 0

    criterion = nn.BCELoss()  # loss

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.view(-1, 1).to(device)
        num_1 += torch.count_nonzero(labels)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.round(outputs)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']

    print(f'test Loss: {epoch_loss} Acc: {epoch_acc} 1: {num_1} 0: {dataset_sizes["test"] - num_1}')


if __name__ == '__main__':
    model = train()
    # model = load()  # to load a saved model

    test(model)
