import torch
import time
from torch import optim
from test import test
from train import train
from data_loader import train_loader, test_loader
from torch.optim.lr_scheduler import MultiStepLR
from model import ConvNet

torch.manual_seed(42)

data_path = '../dataset/mnist'
batch_size = 64
epochs = 20
learning_rate = 1e-3

start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

for epoch in range(1, epochs + 1):
    print('Epoch:', epoch)
    train(model, device, optimizer, scheduler, train_loader(data_path, batch_size))
    test(model, device, test_loader(data_path, batch_size))

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')