import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt as distance

SMOOTH = K.epsilon()


def dice_loss(gt, pr, beta=1, class_weights=1.):
    index = _f_index(gt, pr, beta)
    return (1 - index) * class_weights


def jaccard_loss(gt, pr, beta=1, class_weights=1.):
    index = _iou_index(gt, pr)
    return (1 - index) * class_weights


def tversky_loss(gt, pr, alpha=0.45, beta=0.55, class_weights=1.):
    index = _tversky_index(gt, pr, alpha, beta) * class_weights
    return 1.0 - tf.reduce_mean(index)


def focal_tversky_loss(
        gt, pr, alpha=0.45, beta=0.55, gamma=2.5, class_weights=1.):
    index = _tversky_index(gt, pr, alpha, beta) * class_weights
    loss = K.pow((1.0 - index), gamma)
    return tf.reduce_mean(loss)


def categorical_focal_loss(gt, pr, gamma=2.0, alpha=0.25, class_weights=1.):
    pr = tf.clip_by_value(pr, SMOOTH, 1.0 - SMOOTH)
    loss = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr)) * class_weights
    return tf.reduce_mean(loss)


def log_cosh_dice_loss(gt, pr, beta=1, class_weights=1.):
    x = dice_loss(gt, pr, beta, class_weights)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def log_cosh_tversky_loss(gt, pr, alpha=0.3, beta=0.7, class_weights=1.):
    x = tversky_loss(gt, pr, alpha, beta, class_weights)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def log_cosh_focal_tversky_loss(gt, pr, alpha=0.3, beta=0.7, gamma=1.3, class_weights=1.):
    x = focal_tversky_loss(gt, pr, alpha, beta, gamma, class_weights)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def flooding(loss, b=0.02):
    """Flooding
    arXiv: https://arxiv.org/pdf/2002.08709.pdf
    b is flooding level {0.00, 0.01, 0.02, ..., 0.20}
    """
    return tf.math.abs(loss - b) + b


def rvd_loss(gt, pr, class_weights=1.):
    index = _rvd_index(gt, pr) * class_weights
    return tf.reduce_mean(index)


def surface_loss(gt, pr):
    gt_dist_map = tf.py_function(func=_calc_dist_map_batch,
                                     inp=[gt],
                                     Tout=tf.float32)
    multipled = pr * gt_dist_map
    return tf.reduce_mean(multipled)


def _tp_fp_fn(gt, pr):
    pr = tf.clip_by_value(pr, SMOOTH, 1 - SMOOTH)
    reduce_axes = [0, 1, 2]
    tp = tf.reduce_sum(gt * pr, axis=reduce_axes)
    fp = tf.reduce_sum(pr, axis=reduce_axes) - tp
    fn = tf.reduce_sum(gt, axis=reduce_axes) - tp

    return tp, fp, fn


def _f_index(gt, pr, beta=1):
    tp, fp, fn = _tp_fp_fn(gt, pr)
    intersection = (1 + beta ** 2) * tp + SMOOTH
    summation = (1 + beta ** 2) * tp + beta ** 2 * fn + fp + SMOOTH
    return intersection / summation


def _iou_index(gt, pr):
    tp, fp, fn = _tp_fp_fn(gt, pr)
    intersection = tp + SMOOTH
    union = tp + fn + fp + SMOOTH
    return intersection / union


def _tversky_index(gt, pr, alpha, beta):
    tp, fp, fn = _tp_fp_fn(gt, pr)
    return (tp + SMOOTH) / (tp + alpha * fp + beta * fn + SMOOTH)


def _rvd_index(gt, pr):
    tp, fp, fn = _tp_fp_fn(gt, pr)
    v_label = tp + fn + SMOOTH
    v_infer = tp + fp
    return abs( (v_infer - v_label) / v_label )


def _calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def _calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([_calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)
