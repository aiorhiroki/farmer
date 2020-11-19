import tensorflow as tf
from tensorflow.keras import backend as K

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
    x = focal_tversky_loss(gt, pr, alpha, beta, gamma, class_weights=)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
    

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
    return (tp + SMOOTH) / (tp + alpha*fp + beta*fn + SMOOTH)
