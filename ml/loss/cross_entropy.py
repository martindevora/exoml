from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import tensorflow as tf


def cross_entropy_with_dca_loss(y_true, y_pred, alpha=1., beta=10.):
    """
    Loss function applying penalization to loss decrements when accuracy keeps flatten from Liang et al. (2020)
    https://arxiv.org/abs/2009.04057
    :param y_true: the real labels values (typically one hot encoded)
    :param y_pred: the predicted probabilities
    :param alpha: the alpha corrector for the categorical cross entropy loss
    :param beta: the beta corrector for the dca (difference between confidence and accuracy)
    :return: the computed loss
    """
    ce = CategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    predictions = tf.math.argmax(y_pred, axis=1)
    confidences = tf.reduce_max(y_pred, axis=1)
    mean_conf = tf.reduce_mean(confidences)
    acc_m = CategoricalAccuracy()
    acc_m.update_state(y_true, y_pred)
    acc = acc_m.result().numpy()
    dca = tf.abs(mean_conf - acc)
    loss = alpha * ce + beta * dca
    return loss
