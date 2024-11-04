import torch
import numpy as np
# from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from model import myLeNet, LeNet
from utils import *

batch_size = 128
momentum = 0.5
EPOCH = 100   
BATCH_SIZE = 512      
LR = 0.001        
size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# net = myLeNet
net = LeNet
processes = []



def init_processes(rank, size, device_type, learning_rate, batch_size, epochs, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '25678'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, device_type, learning_rate, batch_size, epochs)
    cleanup()

def partition_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform,download=True)  # 本地没有就加上download=True
    # test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=True)  # train=True训练集，=False测试集
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # TODO: 第二个参数名字是size，分配的时候却变成了比例。现在只是能跑，到时候统一一下好了。
    partitioner = DataPartitioner(train_dataset, [0.5,0.5]) 
    partitioned_data = partitioner.use(dist.get_rank())  # 根据当前进程的rank获得子集

    # 使用DataLoader加载分区数据
    train_loader = torch.utils.data.DataLoader(partitioned_data, batch_size=batch_size, shuffle=True)
    return train_loader


def run(rank, size, device_type, learning_rate, batch_size, epochs):
    torch.manual_seed(1234)
    train_loader = partition_dataset(batch_size)
    # test_loader = get_test_loader(batch_size)
    device = torch.device(f"{device_type}:{rank}")
    print(f"device: {device}")
    
    # 模型定义
    model = net().to(device)
    
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate)

    for epoch in range(EPOCH):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward + backward
            outputs = ddp_model(inputs)
            # loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                    % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

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

# 训练
if __name__ == "__main__":
    # DDP:
    torch.multiprocessing.set_start_method('spawn')
    for rank in range(size):
        p = Process(target=init_processes, args=(
            rank, size, device, LR, batch_size, EPOCH, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

