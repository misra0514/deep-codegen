import torch.nn as nn
import torch
from pytorch_apis import gspmmv
from graphpy import graph_t

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.weight = torch.rand(in_feats, out_feats)
        # self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)


    def forward(self, g, input):
        res = gspmmv(g, input,5,1, False, False  ,'cuda')
        res = res*self.weight
        # g.ndata['h'] = feature
        # g.update_all(gcn_msg, gcn_reduce)
        # g.apply_nodes(func=self.apply_mod)
        return res
    
