
import numpy as np
import torch
import torch.nn as nn
# import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.nn.init as init
import math

sys.path.append("/scratch/yguo25/files/nnsys/deep-codegen")
from pytorch_apis import linear


class mylinear(nn.Module):
    def __init__(self, inputsize, outputsize, device='cuda',bias=True):
        super().__init__()
        self.in_s = inputsize
        self.out_s = outputsize

        # factory_kwargs = {'device': device}
        # self.in_features = in_features
        # self.out_features = out_features
        self.device = device
        self.weight = torch.nn.Parameter(torch.empty((self.in_s, self.out_s)).to(device))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_s)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x):
        result = linear(x, self.weight, x.shape[0], self.out_s ,self.device)
        if self.bias is not None:
            result += self.bias
        return result
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class myLeNet(nn.Module):
    def __init__(self):
        super(myLeNet, self).__init__()
        self.fc1 = nn.Sequential(
            mylinear(28*28, 300,'cuda'),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            mylinear(300, 100,'cuda'),
            nn.ReLU()
        )
        self.fc3 = mylinear(100, 10,'cuda')

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    