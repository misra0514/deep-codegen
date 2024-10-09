import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

batch_size = 128
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform,download=True)  # 本地没有就加上download=True
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


EPOCH = 1   #遍历数据集次数
BATCH_SIZE = 512      #批处理尺寸(batch_size)
LR = 0.001        #学习率



from model import DNN, LeNet

# model = DNN()
model = LeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

# # Construct loss and optimizer ------------------------------------------------------------------------------
# criterion = torch.nn.CrossEntropyLoss()  
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

net = LeNet().to(device)
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# 训练
if __name__ == "__main__":
    with torch.profiler.profile( activities=[ 
    torch.profiler.ProfilerActivity.CPU, 
    torch.profiler.ProfilerActivity.CUDA]) as prof:
        for epoch in range(EPOCH):
            sum_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 每训练100个batch打印一次平均loss
                sum_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %d] loss: %.03f'
                        % (epoch + 1, i + 1, sum_loss / 100))
                    sum_loss = 0.0
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
            
            # TEST
            # with torch.no_grad():
            #     correct = 0
            #     total = 0
            #     for data in test_loader:
            #         images, labels = data
            #         images, labels = images.to(device), labels.to(device)
            #         outputs = net(images)
            #         # 取得分最高的那个类
            #         _, predicted = torch.max(outputs.data, 1)
            #         total += labels.size(0)
            #         correct += (predicted == labels).sum()
            #     print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
