import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data) #前向传播，将每一张28*28像素的图像映射成10个数字，最终第几个数字的值最大这个图片就是手写的几
        loss = F.nll_loss(output, target) #计算当前损失，即衡量按照当前的模型对图像进行分类得到的结果和目标差多少
        optimizer.zero_grad() #清除累积梯度，PyTorch中默认是会累积梯度的，累积梯度对于RNN网络计算很友好
        loss.backward() #进行反向传播，通过链式法则求梯度
        optimizer.step() #进行梯度更新，根据学习率设置的迭代步长进行梯度下降
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))