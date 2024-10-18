import dgl
import argparse
import torch
import scipy.sparse as sp





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

  print(col)

  print(row)



  numlist = torch.arange(col.size(0), dtype=torch.int32)



  adj_csr = sp.csr_matrix((numlist.numpy(), (row, col)), shape=(g.num_nodes(), g.num_nodes()))



  row_ptr=torch.from_numpy(adj_csr.indptr)

  col_ind=torch.from_numpy(adj_csr.indices)



  print(row_ptr)

  print(col_ind)





  features=g.ndata['feat']

  labels=g.ndata['label']

  n_feats=features.shape[1]

  n_classes=data.num_labels

  train_mask = g.ndata['train_mask']

  test_mask = g.ndata['test_mask']

   

  print('row count', row_ptr.size())

  print('col index', col_ind.size())

  print('features', features.size())

  print('labels',  labels.size())

  print('train_mask', train_mask.size(), train_mask)

  print('test_maks', test_mask.size(), test_mask)

