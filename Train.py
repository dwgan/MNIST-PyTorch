import torch.nn.functional as F

def train(model, device, optimizer, scheduler, data_loader):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #清除累积梯度，PyTorch中默认是会累积梯度的，累积梯度对于RNN网络计算很友好
        output = model(data) #前向传播，将每一张28*28像素的图像映射成10个数字，最终第几个数字的值最大这个图片就是手写的几
        loss = F.nll_loss(output, target) #计算当前损失，即衡量按照当前的模型对图像进行分类得到的结果和目标差多少
        loss.backward() #进行反向传播，通过链式法则求梯度
        optimizer.step() #进行梯度更新，根据学习率设置的迭代步长进行梯度下降
        scheduler.step()

        if i % 100 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i * len(data), len(data_loader.dataset),
                100. * i / len(data_loader), loss.item()))