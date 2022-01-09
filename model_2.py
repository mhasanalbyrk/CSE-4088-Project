import numpy as np
import torch
import torchvision

from modelUtil import create_model, predict_local, predict_local_2

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

if __name__ == "__main__":
    model_2 = create_model()
    model_2.load_state_dict(torch.load('model/model.pt'))

    model_1 = torch.load("my_mnist_model.pt")
    model_1.eval()
# model_1 = get_model()
# torch.save(model_1.state_dict(), "model/model.pt")

    device = torch.device("cpu")

    path = "0.png"
    for i in range(10):
        pred = predict_local(path, model_1, device)
        pred_idx = np.argmax(pred)
        print(f'Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %')

    for i in range(10):
        pred = predict_local_2(path, model_2, device)
        pred_idx = np.argmax(pred)
        print(f'Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %')
