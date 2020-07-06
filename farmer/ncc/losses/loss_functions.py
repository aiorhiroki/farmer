from keras import backend
import tensorflow as tf


def _tversky_index(y_true, y_pred, alpha, beta):
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    reduce_axes = [0, 1, 2]
    tp = backend.sum(y_true * y_pred, axis=reduce_axes)
    fp = backend.sum(y_pred, axis=reduce_axes) - tp
    fn = backend.sum(y_true, axis=reduce_axes) - tp
    return (tp + eps) / (tp + alpha*fp + beta*fn + eps)

def focal_tversky_loss(alpha=0.45, beta=0.55, gamma=2., **kwargs):
    gamma = tf.clip_by_value(gamma, 1.0, 3.0)
    def loss(y_true, y_pred):
        index =_tversky_index(y_true, y_pred, alpha, beta)
        loss = backend.pow((1.0 - index), (1.0 / gamma))
        return backend.mean(loss)
    return loss


