from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageOps
from torch import nn

from datasetUtil import get_dataset
from train import train

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def create_model():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model


def predict(path, model, device):
    r = requests.get(path)
    with BytesIO(r.content) as f:
        img = Image.open(f).convert(mode="L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()


# def train(model, dataset, val_dataset, numb_epoch=3, lr=1e-3, device="cpu"):

def get_model():
    device = torch.device("cpu")
    dataset, val_dataset = get_dataset(128)
    model = train(create_model(), dataset, val_dataset, device=device)
    return model


def predict_local_2(path, model, device):
    img = Image.open(path).convert(mode="L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    # x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()

def predict_local_3(img, model, device):
    #img = Image.open(path).convert(mode="L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    x = (255 - np.expand_dims(img, -1)) / 255.
    # x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(pred, dim=-1).cpu().numpy()

def predict_local(path, model, device):
    img = Image.open(path).convert(mode="L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = T(img)
    img = img.view(1, 784)
    # x = (255 - np.expand_dims(np.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(img)
        return F.softmax(pred, dim=-1).cpu().numpy()
