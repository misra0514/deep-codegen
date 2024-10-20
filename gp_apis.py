import torch as th
import torch.utils.dlpack
import graphpy as gpk
def gp_linear(X, W, dim1_0, dim1_1, device0):
    X_dl = th.utils.dlpack.to_dlpack(X)
    W_dl = th.utils.dlpack.to_dlpack(W)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.linear(X_dl, W_dl, res_dl1)
    return res1
def gp_gspmmv(graph, input1, dim1_0, dim1_1, reverse, norm, device0):
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    res1 = th.zeros(dim1_0, dim1_1, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    gpk.gspmmv(graph, input1_dl, res_dl1, reverse, norm)
    return res1
