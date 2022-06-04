from keras.activations import softmax
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.metrics import CategoricalAccuracy, BinaryAccuracy
import tensorflow as tf


def categorical_cross_entropy_with_dca_loss(y_true, y_pred, alpha=1., beta=10., from_logits=False,
                                            activation_function=softmax):
    """
    DCA loss applied to categorical classification
    :param y_true: the real labels values (typically one hot encoded)
    :param y_pred: the predicted probabilities
    :param alpha: the alpha corrector for the categorical cross entropy loss
    :param beta: the beta corrector for the dca (difference between confidence and accuracy)
    :param from_logits: whether use weights or probabilities
    :param activation_function: the activation function to be applied if from_logits=True
    :return: the dca loss
    """
    return dca_loss(y_true, y_pred, CategoricalCrossentropy(from_logits=from_logits), CategoricalAccuracy(), alpha,
                    beta, from_logits, activation_function)

def binary_cross_entropy_with_dca_loss(y_true, y_pred, alpha=1., beta=10., from_logits=False,
                                       activation_function=softmax):
    """
    DCA loss applied to binary classification
    :param y_true: the real labels values (typically one hot encoded)
    :param y_pred: the predicted probabilities
    :param alpha: the alpha corrector for the categorical cross entropy loss
    :param beta: the beta corrector for the dca (difference between confidence and accuracy)
    :param from_logits: whether use weights or probabilities
    :param activation_function: the activation function to be applied if from_logits=True
    :return: the dca loss
    """
    return dca_loss(y_true, y_pred, BinaryCrossentropy(from_logits=from_logits), BinaryAccuracy(), alpha, beta,
                    from_logits, activation_function)

def dca_loss(y_true, y_pred, loss, accuracy, alpha=1., beta=10., from_logits=False, activation_function=softmax):
    """
    Loss function applying penalization to loss decrements when accuracy keeps flatten from Liang et al. (2020)
    https://arxiv.org/abs/2009.04057 (https://stackoverflow.com/questions/68755788/keras-version-of-the-combined-cross-entropy-and-calibration-loss)
    :param y_true: the real labels values (typically one hot encoded)
    :param y_pred: the predicted probabilities
    :param alpha: the alpha corrector for the categorical cross entropy loss
    :param beta: the beta corrector for the dca (difference between confidence and accuracy)
    :param from_logits: whether use weights or probabilities
    :param activation_function: the activation function to be applied if from_logits=True
    :return: the computed loss
    """
    loss_value = loss(y_true, y_pred)
    pred_values = y_pred
    if from_logits and activation_function is not None:
        pred_values = softmax(y_true, axis=1)
    predictions = tf.math.argmax(pred_values, axis=1)
    confidences = tf.reduce_max(pred_values, axis=1)
    mean_conf = tf.reduce_mean(confidences)
    accuracy.update_state(pred_values, y_pred)
    acc = accuracy.result().numpy()
    dca = tf.abs(mean_conf - acc)
    loss_value = alpha * loss_value + beta * dca
    return loss_value
