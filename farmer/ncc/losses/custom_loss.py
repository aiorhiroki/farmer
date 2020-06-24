from keras import backend as K
import tensorflow as tf

def tversky_index(y_true, y_pred, alpha=0.3, beta=0.7):
    # 参考サイト：http://ni4muraano.hatenablog.com/entry/2018/04/02/223236

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    false_positive = K.sum((1.0 - y_true) * y_pred)
    false_negative = K.sum(y_true * (1.0 - y_pred))
    return intersection / (intersection + alpha*false_positive + beta*false_negative)

def tversky_loss(y_true, y_pred):
    return 1.0 - tversky_index(y_true, y_pred)