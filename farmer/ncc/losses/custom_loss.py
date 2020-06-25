from keras import backend as K
import tensorflow as tf


def tversky_loss(alpha=0.3, beta=0.7):

    eps = tf.keras.backend.epsilon()

    def tversky_index(y_true, y_pred, alpha, beta):
        true_positive = y_true * y_pred
        false_positive = (1.0 - y_true) * y_pred
        false_negative = y_true * (1.0 - y_pred)
        for i in range(3):
            true_positive = K.sum(true_positive, axis=0)
            false_positive = K.sum(false_positive, axis=0)
            false_negative = K.sum(false_negative, axis=0)
        return (true_positive + eps) / (true_positive + alpha*false_positive + beta*false_negative + eps)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        loss =tversky_index(y_true, y_pred, alpha, beta)
        return tf.reduce_mean(loss)

    return loss