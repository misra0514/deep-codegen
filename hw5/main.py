import dgl
import argparse
import torch
import scipy.sparse as sp
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

  optimizer = torch.optim.Adam((model.parameters()), lr=0.1, weight_decay=1e-3) 
  

  # TRAIN
  # for epoch in tqdm(range(1000)):
  x = []
  y1 = []
  y2 = []
  fig, ax1 = plt.subplots()
  plt.xticks()
  for epoch in range(1000):
    out =  model(g.ndata['feat'].cuda())
    label = g.ndata['label'].cuda()
    out = torch.squeeze(out)

    loss = (out-label).abs()

    loss = loss.sum()
    acc = label.sum()
    acc = loss.item()/ acc.item()
    acc = 1-acc
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('Epoch', epoch, 'loss  ', loss.item(),'  Acc:', acc)
    if(epoch % 50 ==0):
      x.append(epoch)
      y1.append(loss.item())
      y2.append(acc)


  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss', color='tab:red')
  ax1.plot(x, y1, color='tab:red', label='Loss')
  ax1.tick_params(axis='y', labelcolor='tab:red')

  ax2 = ax1.twinx()  
  ax2.set_ylabel('Accuracy', color='tab:blue')
  ax2.plot(x, y2, color='tab:blue', label='Accuracy')
  ax2.tick_params(axis='y', labelcolor='tab:blue')

  plt.title('Training Loss and Accuracy over ' + args.dataset)
  fig.legend(loc='upper left')
  fig.tight_layout()
  plt.savefig('./result/'+args.dataset+'.png')

