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
def gp_gspmmv(graph, X1, dim_0, dim_1, reverse, norm):
    X1_dl = tf.experimental.dlpack.to_dlpack(X1)
    #declare the output tensor here
    res = tf.zeros([dim_0, dim_1])
    res_dl = tf.experimental.dlpack.to_dlpack(res)
    gpk.gspmmv(graph, X1_dl, res_dl, reverse, norm)
    return res
