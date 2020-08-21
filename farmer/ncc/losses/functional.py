import segmentation_models
from segmentation_models.base import Loss

import tensorflow as tf
from tensorflow.keras import backend as K


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, class_weights=None):
        super().__init__(name='tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return _tversky_loss(
            y_true=gt, 
            y_pred=pr, 
            alpha=self.alpha, 
            beta=self.beta, 
            class_weights=self.class_weights
        )


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, gamma=2.5, class_weights=None):
        super().__init__(name='tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return _focal_tversky_loss(
            y_true=gt, 
            y_pred=pr, 
            alpha=self.alpha, 
            beta=self.beta, 
            gamma=self.gamma, 
            class_weights=self.class_weights
        )


def _tversky_index(y_true, y_pred, alpha, beta):
    eps = K.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    reduce_axes = [0, 1, 2]
    tp = tf.reduce_sum(y_true * y_pred, axis=reduce_axes)
    fp = tf.reduce_sum(y_pred, axis=reduce_axes) - tp
    fn = tf.reduce_sum(y_true, axis=reduce_axes) - tp
    return (tp + eps) / (tp + alpha*fp + beta*fn + eps)


def _tversky_loss(y_true, y_pred, alpha=0.45, beta=0.55, class_weights=1., **kwargs):
    index = _tversky_index(y_true, y_pred, alpha, beta) * class_weights
    return 1.0 - tf.reduce_mean(index)


def _focal_tversky_loss(y_true, y_pred, alpha=0.45, beta=0.55, gamma=2.5, class_weights=1., **kwargs):
    gamma = tf.clip_by_value(gamma, 1.0, 3.0)
    index =_tversky_index(y_true, y_pred, alpha, beta) * class_weights
    loss = K.pow((1.0 - index), (1.0 / gamma))
    return K.mean(loss)