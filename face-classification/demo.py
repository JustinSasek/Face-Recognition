import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from model import model
from cv2 import VideoCapture
import os


WEBCAM_NUM = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    model = model()

    model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model.pt')))
    model.eval()  # Set model to evaluate mode

    # Transformations done on the dataset during pretraining on imagenet
    img_transform = T.Compose([T.ToTensor(),
                               T.Resize(256),
                               T.CenterCrop(224),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])

    cam = VideoCapture(WEBCAM_NUM)

    while True:
        result, image = cam.read()
        if result:
            img = img_transform(image)  # resize, crop, normalize, etc.
            inputs = torch.unsqueeze(img.to(device), 0)  # send image to device and add batch dim

            # forward
            outputs = model(inputs)
            preds = torch.round(outputs)

            print(preds.cpu().detach().numpy()[0][0] == 1)
        else:
            print('Error: Could not capture image')
