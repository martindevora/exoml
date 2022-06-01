import tensorflow as tf
import numpy as np
from sklearn.metrics import auc
import tensorflow.keras.backend as K

@tf.function
def confusion_matrix_values(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_pred_label = tf.reshape(y_pred_label, tf.shape(y_pred_label)[0])
    y_true_label = tf.reshape(y_true_label, tf.shape(y_true_label)[0])
    real_positive_indexes_mask = tf.math.greater(y_true_label, 0)
    predicted_positive_indexes_mask = tf.math.greater(y_pred_label, threshold)
    positive_indexes_intersect_mask = tf.math.logical_and(predicted_positive_indexes_mask, real_positive_indexes_mask)
    predicted_positives = tf.where(predicted_positive_indexes_mask).shape.as_list()[0]
    predicted_positives_in_true_set = tf.where(positive_indexes_intersect_mask).shape.as_list()[0]
    tp = predicted_positives_in_true_set / (predicted_positives + tf.keras.backend.epsilon())
    real_negative_indexes_mask = tf.math.equal(y_true_label, 0)
    predicted_negative_indexes_mask = tf.math.less_equal(y_pred_label, threshold)
    negative_indexes_intersect_mask = tf.math.logical_and(predicted_negative_indexes_mask, real_negative_indexes_mask)
    predicted_negatives = tf.where(predicted_negative_indexes_mask).shape.as_list()[0]
    predicted_negatives_in_true_set = tf.where(negative_indexes_intersect_mask).shape.as_list()[0]
    correct_predictions = predicted_positives_in_true_set + predicted_negatives_in_true_set
    total_predictions = y_true.shape.as_list()[0]
    accuracy = correct_predictions / (total_predictions + K.epsilon())
    error_rate = 1 - accuracy
    #TODO decide what values to be returned when no positives are detected
    if predicted_positives == 0:
        return [[np.nan, np.nan], [np.nan, np.nan]], accuracy, error_rate, np.nan, np.nan, np.nan
    else:
        tn = predicted_negatives_in_true_set / (predicted_negatives + tf.keras.backend.epsilon())
        fp = (predicted_positives - predicted_positives_in_true_set) / (predicted_positives + K.epsilon())
        fn = (predicted_negatives - predicted_negatives_in_true_set) / (predicted_negatives + K.epsilon())
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return [[tp, fp], [fn, tn]], accuracy, error_rate, precision, recall, f1_score


@tf.function
def area_under_curve(y_true, y_pred, label_index=3, thresholds=100, method='roc'):
    confusion_matrixes = np.empty((thresholds, 2, 2))
    accuracies = np.zeros(thresholds)
    error_rates = np.zeros(thresholds)
    precisions = np.zeros(thresholds)
    recalls = np.zeros(thresholds)
    f1_scores = np.zeros(thresholds)
    threshold_values = np.flip(np.linspace(0, 1, thresholds))
    for index, threshold in enumerate(threshold_values):
        confusion_matrixes[index], accuracies[index], error_rates[index], precisions[index], recalls[index], \
            f1_scores[index] = confusion_matrix_values(y_true, y_pred, label_index=label_index, threshold=threshold)
    x = recalls[~np.isnan(recalls)]
    y = precisions[~np.isnan(precisions)]
    return np.trapz(y, x)

class F1History(tf.keras.callbacks.Callback):
    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):
        logs['F1_score_train'] = float('-inf')
        x_train, y_train = self.train[0], self.train[1]
        y_pred = self.model.predict(x_train)
        confusion_matrix, accuracy, error_rate, precision, recall, f1_score = self.confusion_matrix(y_train, y_pred)
        logs['F1_score_train'] = np.round(f1_score, 5)
        if self.validation:
            logs['F1_score_val'] = float('-inf')
            x_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = self.model.predict(x_valid)
            confusion_matrix, accuracy, error_rate, precision, recall, f1_score = self.confusion_matrix(y_valid, y_val_pred)
            logs['F1_score_val'] = np.round(f1_score, 5)
