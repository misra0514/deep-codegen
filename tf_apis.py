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

