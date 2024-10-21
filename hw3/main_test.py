import dgl
import argparse
import torch
import scipy.sparse as sp
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

import sys
sys.path.append("/mnt/data/home/yguo/projects/sys4NN/deep-codegen")
from pytorch_apis import linear
from graphpy import graph_t
import graphpy
import numpy as np
import itertools

from tqdm import tqdm

import model2



if __name__ == '__main__':
  # 1 Prepare Dataset && graph structure
  parser = argparse.ArgumentParser(description='downaload DGL graph data for CSCI 780-02')
  parser.add_argument("--dataset",type=str,default="cora")
  args = parser.parse_args()

  if args.dataset == 'cora':
    data = dgl.data.CoraGraphDataset()
  elif args.dataset == 'citeseer':
    data = dgl.data.CiteseerGraphDataset()
  elif args.dataset == 'pubmed':
    data = dgl.data.PubmedGraphDataset()
  elif args.dataset == 'reddit':
    data = dgl.data.RedditDataset()
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

  g = data[0]
  col,row=g.edges(order='srcdst')
  torch.set_printoptions(edgeitems=100)
  numlist = torch.arange(col.size(0), dtype=torch.int32)
  # adjcsr 仅仅是临界矩阵
  adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
  row_ptr=torch.from_numpy(adj_csr.indptr)
  col_ind=torch.from_numpy(adj_csr.indices)
  numlist = torch.arange(col.size(0), dtype=torch.int32)
  # # TODO: col_ind 可能还有点问题，这个数组是ppt里的 DEST Vertex吗..?

  Vnum = row_ptr.size()[0]-1
  Enum = col_ind.size()[0]

  gra = graphpy.init_graph(Vnum ,Enum,adj_csr.indices, adj_csr.indices.astype(np.int32) ,adj_csr.indptr.astype(np.int32) )
  model = model2.TwoLayerGCN(in_feats=g.ndata['feat'].shape[1],hidden_feats=32, out_feats=1 , graph=gra, device='cuda')
  model.to('cuda')
  # print(model.parameters())
  # for name, param in model.named_parameters():
  #     print(f"参数名称: {name}")
  #     print(f"参数值:\n{param}")
  #     print(f"参数的形状: {param.shape}")
  #     print()

  optimizer = torch.optim.Adam((model.parameters()), lr=0.01) 
  scaler = GradScaler()
  


  # TRAIN
  # for epoch in tqdm(range(1000)):
  for epoch in range(1000):
    out =  model(g.ndata['feat'].cuda())
    label = g.ndata['label'].cuda()
    out = torch.squeeze(out)

    # print(torch.squeeze(out).shape)
    # print(label.shape)
    # loss = F.cross_entropy(out, label)
    loss = (out-label).abs()
    loss = loss.sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    print('epoch', epoch, 'loss', loss.item())
