import torch
import torch.optim as optim
from model import ConvNet
from train import train
from test import test
from data_loader import train_loader, test_loader

batch_size = 512
epochs = 20
learning_rate = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train(model, device, train_loader(batch_size), optimizer, epoch)
    test(model, device, test_loader(batch_size))