from keras import backend
import tensorflow as tf

import segmentation_models
from segmentation_models import Unet, PSPNet, FPN

def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x

def get_reduce_axes(per_image=False):
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes

def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs

def tversky_loss(alpha=0.5, beta=0.5):
    eps = tf.keras.backend.epsilon()
    def tversky_index(y_true, y_pred, alpha, beta):
        gt, pr = gather_channels(y_true, y_pred, indexes=None)
        axes = get_reduce_axes()
        tp = backend.sum(gt * pr, axis=axes)
        fp = backend.sum(pr, axis=axes) - tp
        fn = backend.sum(gt, axis=axes) - tp
        return (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        index =tversky_index(y_true, y_pred, alpha, beta)
        return 1.0 - backend.mean(index)
    return loss