import tensorlow as tf
import kernel as gpk
def gp_linear(X, W, dim1;_0, dim1;_1):
    X_dl = tf.experimental.dlpack.to_dlpack(X)
    W_dl = tf.experimental.dlpack.to_dlpack(W)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.linear(X_dl, W_dl, res_dl)
    return res
