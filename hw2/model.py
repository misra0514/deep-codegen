
import numpy as np
import torch
import torch.nn as nn

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



class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers =  [1,10, 10 ,1]   #网络每一层的神经元个数，[1,10,1]说明只有一个隐含层，输入的变量是一个，也对应一个输出。如果是两个变量对应一个输出，那就是[2，10，1]
        self.layer1 = mylinear(layers[0],layers[1])  #用torh.nn.Linear构建线性层，本质上相当于构建了一个维度为[layers[0],layers[1]]的矩阵，这里面所有的元素都是权重
        self.layer2 = mylinear(layers[1],layers[2])
        self.layer3 = mylinear(layers[2],layers[3])
        self.elu = nn.ELU()      
    def forward(self,d):
        d1 = self.layer1(d)
        d1 = self.elu(d1)
        d2 = self.layer2(d1)
        d2 = self.elu(d2)
        d2 = self.layer3(d2)
        return d2
    
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