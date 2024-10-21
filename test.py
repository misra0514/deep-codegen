import torch
from graphpy import graph_t
import graphpy
import dgl
import time
from pytorch_apis import gspmmv,linear
import scipy.sparse as sp
import  numpy as np

# # TEST 1
# # nodeNum=8
# # edgeNum=11
# offset = torch.tensor([0,3,5,7,9,12]).numpy().astype(np.int32)
# # offset = torch.from_numpy(offset)
# col =  torch.tensor([1,2,4,0,4,0,3,2,4,0,1,3]).numpy()
# # col = torch.from_numpy(col)
# # input = torch.range(0,5)
# input = torch.ones([5,1]).cuda()

# gra = graphpy.init_graph(5 ,12,col, col ,offset )
# res = gspmmv(gra, input,5,1, False, False  ,'cuda')
# print(res)

# # TEST2 : Cora

data = dgl.data.CoraGraphDataset()
g = data[0]
col,row=g.edges(order='srcdst')
# torch.set_printoptions(edgeitems=100)
torch.set_printoptions(profile="full")
# print(col)
# print(row)

numlist = torch.arange(col.size(0), dtype=torch.int32)
# adjcsr 仅仅是临界矩阵
adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
row_ptr=torch.from_numpy(adj_csr.indptr)
col_ind=torch.from_numpy(adj_csr.indices)
# # TODO: col_ind 可能还有点问题，这个数组是ppt里的 DEST Vertex吗..?
# print(col_ind)
# pass

Vnum = row_ptr.size()[0]-1
Enum = col_ind.size()[0]

gra = graphpy.init_graph(Vnum ,Enum,adj_csr.indices, adj_csr.indices.astype(np.int32) ,adj_csr.indptr.astype(np.int32) )
# # time.sleep(10)

res = gspmmv(gra, torch.ones([Vnum,1]).cuda(),Vnum,1, False, False  ,'cuda')
print(res.shape)
print(res)