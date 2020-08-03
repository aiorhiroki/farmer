import segmentation_models
from segmentation_models.base import Loss
from segmentation_models.losses import (
    DiceLoss, JaccardLoss, CategoricalFocalLoss, CategoricalCELoss
)

import tensorflow as tf


segmentation_models.set_framework('tf.keras')


def dice_loss(beta=1, class_weights=None, class_indexes=None, **kwargs):
    loss = DiceLoss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return loss


def jaccard_loss(class_weights=None, class_indexes=None, per_image=False, **kwargs):
    loss = JaccardLoss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return loss


def categorical_focal_loss(alpha=0.25, gamma=2., class_indexes=None, **kwargs):
    loss = CategoricalFocalLoss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    return loss


def categorical_crossentropy_loss(class_weights=None, class_indexes=None, **kwargs):
    loss = CategoricalCELoss(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return loss


def cce_dice_loss(beta=1, class_weights=None, class_indexes=None, **kwargs):
    cce = categorical_crossentropy(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    dl = dice_loss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return cce + dl


def cce_jaccard_loss(class_weights=None, class_indexes=None, per_image=False, **kwargs):
    cce = categorical_crossentropy(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    jl = jaccard_loss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return cce + jl


def categorical_focal_dice_loss(alpha=0.25, beta=1, gamma=2.,
                                class_weights=None, class_indexes=None, **kwargs):
    cfl = categorical_focal_loss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    dl = dice_loss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return cfl + dl


def categorical_focal_jaccard_loss(alpha=0.25, gamma=2., class_weights=None,
                                   class_indexes=None, per_image=False, **kwargs):
    cfl = categorical_focal_loss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    jl = jaccard_loss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return cfl + jl


def _tversky_index(y_true, y_pred, alpha, beta):
    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    reduce_axes = [0, 1, 2]
    tp = tf.reduce_sum(y_true * y_pred, axis=reduce_axes)
    fp = tf.reduce_sum(y_pred, axis=reduce_axes) - tp
    fn = tf.reduce_sum(y_true, axis=reduce_axes) - tp
    return (tp + eps) / (tp + alpha*fp + beta*fn + eps)

def focal_tversky_loss(alpha=0.45, beta=0.55, gamma=2.5, **kwargs):
    gamma = tf.clip_by_value(gamma, 1.0, 3.0)
    def loss(y_true, y_pred):
        index =_tversky_index(y_true, y_pred, alpha, beta)
        loss = backend.pow((1.0 - index), (1.0 / gamma))
        return backend.mean(loss)
    return loss


def tversky_loss(alpha=0.45, beta=0.55, **kwargs):
    def loss(y_true, y_pred):
        index =_tversky_index(y_true, y_pred, alpha, beta)
        return 1.0 - tf.reduce_mean(index)
    return loss
