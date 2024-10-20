import torch
from graphpy import graph_t
import graphpy
import dgl
import time
from pytorch_apis import gspmmv,linear

linear(torch.rand([2,2]).cuda() , torch.rand([2,2]).cuda(),2,2 ,'cuda')

# data = dgl.data.CoraGraphDataset()
# g = data[0]
# col,row=g.edges(order='srcdst')


# # a = graph_t()
# # print(col.shape) #torch.Size([10556])
# col = col.numpy()
# gra = graphpy.init_graph( 100 ,100 , col, col )
# # # time.sleep(10)
# # print(temp.get_vcount()) # 即使sleep也还是200

# gspmmv(gra, torch.rand([2,2]).cuda(),2,2, True, True ,'cuda')