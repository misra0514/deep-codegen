import tensorflow as tf
import gp_apis

def linear(X, W, dim1;_0, dim1;_1, device0):
    @tf.custom_gradient
    def _lambda():
        return linear_real(X, W, dim1;_0, dim1;_1, device0)
    return _lambda()

def linear_real(X, W, dim1;_0, dim1;_1, device0):
    out = gp_apis.gp_linear(X, W, dim1;_0, dim1;_1, device0)
    def grad():
        return gp_apis.gp_linear(X, W, dim1;_0, dim1;_1, device0)
    return out, grad

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    @tf.custom_gradient
    def _lambda(X1):
        return gspmmv_real(graph, X1, dim_0, dim_1, reverse, norm, device0)
    return _lambda(input1)

def gspmmv_real(graph, input1, dim_0, dim_1, reverse, norm, device0):
    out = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, 1, norm, device0)
    def grad(dZ1):
        return gp_apis.gp_gspmmv(graph, dZ1, dim_0, dim_1, 0, norm, device0)
    return out, grad

