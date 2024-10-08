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

# fig = plt.figure()
# for i in range(12):
#     plt.subplot(3, 4, i+1)
#     plt.tight_layout()
#     plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
#     plt.title("Labels: {}".format(train_dataset.train_labels[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()


from model import DNN

model = DNN()

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量

def train(epoch):
    running_loss = 0.0  
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        # if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
    print('[%d, %5d]: loss: %.3f , acc: %.2f %%' (epoch , running_loss / 300, 100 * running_correct / running_total))
    running_loss = 0.0  # 这小批300的loss清零
    running_total = 0
    running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)

        # acc_test = test()
        # acc_list_test.append(acc_test)

    # plt.plot(acc_list_test)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy On TestSet')
    # plt.show()
