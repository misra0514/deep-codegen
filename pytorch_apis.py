import torch as th
import gp_apis

class linear_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, dim1_0, dim1_1, device0):
        # ctx.backward_cache = W #must be implemented
        # ctx.backward_cache = X #must be implemented
        ctx.save_for_backward(W)
        res = gp_apis.gp_linear(X, W, dim1_0, dim1_1, device0)
        return res

    @staticmethod
    def backward(ctx, dZ):
        w = ctx.saved_tensors
        return dZ*w


def linear(X, W, dim1_0, dim1_1, device0):
    return linear_impl.apply(X, W, dim1_0, dim1_1, device0)

