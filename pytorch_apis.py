import torch as th
import gp_apis

class linear_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, dim1_0, dim1_1, device0):
        res = gp_apis.gp_linear(X, W, dim1_0, dim1_1, device0)
        ctx.backward_cache = X,W,dim1_0, dim1_1 #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        X, W,dim1_0, dim1_1 = ctx.backward_cache
        dydx = linear_impl.apply(dZ, W, dim1_0, dim1_1, 'cuda')
        X = X.t().contiguous()
        dydw = linear_impl.apply(X, dZ, dim1_0, dim1_1, 'cuda')
        return dydx, dydw, None, None, None

def linear(X, W, dim1_0, dim1_1, device0):
    return linear_impl.apply(X, W, dim1_0, dim1_1, device0)

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, reverse, norm, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0)
        ctx.backward_cache = graph #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        graph = ctx.backward_cache
        res = gp_apis.gp_gspmmv(graph, dZ, dZ.shape[0],  dZ.shape[1], False, False, 'cuda')
        return None, res, None, None, None, None, None

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, reverse, norm, device0)

