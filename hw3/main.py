import dgl
import argparse
import torch
import scipy.sparse as sp
import sys
sys.path.append("/mnt/data/home/yguo/projects/sys4NN/deep-codegen")
from pytorch_apis import linear
from graphpy import graph_t

import model2



if __name__ == '__main__':

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
  # add self loop
  #g = dgl.remove_self_loop(g)
  #g = dgl.add_self_loop(g)

  col,row=g.edges(order='srcdst')

  torch.set_printoptions(edgeitems=100)

  # print(col)
  # print(row)

  numlist = torch.arange(col.size(0), dtype=torch.int32)
  # adjcsr 仅仅是临界矩阵
  adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))
  row_ptr=torch.from_numpy(adj_csr.indptr)
  col_ind=torch.from_numpy(adj_csr.indices)

  # print(row_ptr)
  # print(col_ind)

  # print(adj_csr.shape)
  # print(adj_csr.indptr) # 长度2709，  似乎是offset
  # print(adj_csr.indices) # 感觉上似乎是Dest. Vertex


  # features=g.ndata['feat']
  # labels=g.ndata['label']
  # n_feats=features.shape[1]
  # n_classes=data.num_labels
  # train_mask = g.ndata['train_mask']
  # test_mask = g.ndata['test_mask']

  # print('row count', row_ptr.size()) #2709
  # print('col index', col_ind.size()) # 10556
  # print('features', features.size()) #([2708, 1433])
  # print('labels',  labels.size())
  # print('train_mask', train_mask.size(), train_mask)
  # print('test_maks', test_mask.size(), test_mask)

