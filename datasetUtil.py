import torch
import torchvision

T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
def get_dataset(batch):
    train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
    val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

    train_dl = torch.utils.data.DataLoader(train_data, batch_size =batch)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size = batch)

    return train_dl, val_dl
