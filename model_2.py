import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from modelUtil import get_model, predict, Net

import copy
import numpy as np


T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


if __name__ == "__main__":
    model = get_model()
    torch.save(model.state_dict(), "model/model.pt")

    device = torch.device("cpu")

    # model = Net().to(device)
    # model.load_state_dict(torch.load('model/model.pt'))

    path = "https://c8.alamy.com/comp/HXBRW0/2-red-handwritten-digits-over-white-background-HXBRW0.jpg"
    pred = predict(path, model, device)
    pred_idx = np.argmax(pred)
    print(f'Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %')


