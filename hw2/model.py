
import numpy as np
import torch
import torch.nn as nn
# import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/mnt/data/home/yguo/projects/sys4NN/deep-codegen")
from pytorch_apis import linear
# a = torch.rand(2,2).cuda()
# a = torch.ones([2,2]).cuda()

# a = torch.arange(2, 6, 2).cuda()

# a = torch.tensor([[1,2],[3,4]]).cuda()
# a = torch.ones([2,2]).cuda()
# arr = np.array([[1, 2], [3,4]])  
# a = torch.from_numpy(arr)
# print(a)
# print("----")
# b = torch.ones(2,2).cuda()*2
# print(linear(a,b,2,2,'cuda'))
# print(torch.matmul(a,b))
# X: B*int, W: in * out ---> B*out
class mylinear(nn.Module):
    def __init__(self, inputsize, outputsize):
        super().__init__()
        self.in_s = inputsize
        self.out_s = outputsize
        self.b = torch.rand(1)
        self.w = torch.rand([ outputsize, inputsize])
    def forward(self, x):
        # TODO: use view to transpose
        w_t = self.w.view(self.in_s, self.out_s)
        linear(x, w_t, self.in_s, self.out_s ,'cuda')
        x += self.b
        return x



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2), #padding=2保证输入输出尺寸相同
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            mylinear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            mylinear(120, 84),
            nn.ReLU()
        )
        self.fc3 = mylinear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    