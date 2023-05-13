import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from keras.metrics import TopKCategoricalAccuracy
from tensorflow.python.ops.metrics_impl import precision


@tf.function
def tp_tf(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = y_true
    y_pred_label = y_pred
    real_positive_indexes_mask = tf.math.greater(y_true_label, 0)
    predicted_positive_indexes_mask = tf.math.greater(y_pred_label, threshold)
    positive_indexes_intersect_mask = tf.math.logical_and(predicted_positive_indexes_mask, real_positive_indexes_mask)
    predicted_positives = tf.cast(tf.size(tf.where(predicted_positive_indexes_mask)), tf.float32)
    predicted_positives_in_true_set = tf.cast(tf.size(tf.where(positive_indexes_intersect_mask)), tf.float32)
    tp = predicted_positives_in_true_set / (predicted_positives + tf.keras.backend.epsilon())
    return tp


@tf.function
def fp_tf(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = y_true
    y_pred_label = y_pred
    real_positive_indexes_mask = tf.math.greater(y_true_label, 0)
    predicted_positive_indexes_mask = tf.math.greater(y_pred_label, threshold)
    positive_indexes_intersect_mask = tf.math.logical_and(predicted_positive_indexes_mask, real_positive_indexes_mask)
    predicted_positives = tf.cast(tf.size(tf.where(predicted_positive_indexes_mask)), tf.float32)
    predicted_positives_in_true_set = tf.cast(tf.size(tf.where(positive_indexes_intersect_mask)), tf.float32)
    fp = (predicted_positives - predicted_positives_in_true_set) / (predicted_positives + K.epsilon())
    return fp


@tf.function
def fn_tf(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    real_negative_indexes_mask = tf.math.equal(y_true_label, 0)
    predicted_negative_indexes_mask = tf.math.less_equal(y_pred_label, threshold)
    negative_indexes_intersect_mask = tf.math.logical_and(predicted_negative_indexes_mask, real_negative_indexes_mask)
    predicted_negative_indexes_mask = tf.math.less_equal(y_pred_label, threshold)
    predicted_negatives = tf.cast(tf.size(tf.where(predicted_negative_indexes_mask)), tf.float32)
    predicted_negatives_in_true_set = tf.cast(tf.size(tf.where(negative_indexes_intersect_mask)), tf.float32)
    fn = (predicted_negatives - predicted_negatives_in_true_set) / (predicted_negatives + K.epsilon())
    return fn


@tf.function
def tn_tf(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    real_negative_indexes_mask = tf.math.equal(y_true_label, 0)
    predicted_negative_indexes_mask = tf.math.less_equal(y_pred_label, threshold)
    negative_indexes_intersect_mask = tf.math.logical_and(predicted_negative_indexes_mask, real_negative_indexes_mask)
    predicted_negative_indexes_mask = tf.math.less_equal(y_pred_label, threshold)
    predicted_negatives = tf.cast(tf.size(tf.where(predicted_negative_indexes_mask)), tf.float32)
    predicted_negatives_in_true_set = tf.cast(tf.size(tf.where(negative_indexes_intersect_mask)), tf.float32)
    tn = predicted_negatives_in_true_set / (predicted_negatives + tf.keras.backend.epsilon())
    return tn


@tf.function
def precision_tf(y_true, y_pred, label_index=0, threshold=0.5):
    tp = tp_tf(y_true, y_pred, label_index, threshold)
    fp = fp_tf(y_true, y_pred, label_index, threshold)
    return tp / (tp + fp + K.epsilon())


@tf.function
def recall_tf(y_true, y_pred, label_index=0, threshold=0.5):
    tp = tp_tf(y_true, y_pred, label_index, threshold)
    fn = fn_tf(y_true, y_pred, label_index, threshold)
    return tp / (tp + fn + K.epsilon())


@tf.function
def f1_score_tf(y_true, y_pred, label_index=0, threshold=0.5):
    precision = precision_tf(y_true, y_pred, label_index, threshold)
    recall = recall_tf(y_true, y_pred, label_index, threshold)
    return 2 * precision * recall / (precision + recall + K.epsilon())

@tf.function
def precision_at_k(y_true, y_pred, label_index=0, k=100, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    values, indices = tf.math.top_k(y_pred_label, k=k)
    y_true_label = tf.gather(y_true_label, indices)
    y_pred_label = tf.gather(y_pred_label, indices)
    return precision_tf(y_true_label, y_pred_label, label_index=0, threshold=threshold)

@tf.function
def mean_positive_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    positive_pred_indexes = tf.where(tf.greater(y_pred_label, threshold))
    return tf.reduce_mean(tf.gather(y_pred_label, positive_pred_indexes))

@tf.function
def mean_true_positive_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    tp_pred_indexes = tf.where(tf.logical_and(tf.greater(y_pred_label, threshold), tf.equal(y_true_label, 1)))
    return tf.reduce_mean(tf.gather(y_pred_label, tp_pred_indexes))

@tf.function
def mean_false_positive_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    fp_pred_indexes = tf.where(tf.logical_and(tf.greater(y_pred_label, threshold), tf.equal(y_true_label, 0)))
    return tf.reduce_mean(tf.gather(y_pred_label, fp_pred_indexes))

@tf.function
def mean_negative_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    negative_pred_indexes = tf.where(tf.less(y_pred_label, threshold))
    return tf.reduce_mean(tf.gather(y_pred_label, negative_pred_indexes))

@tf.function
def mean_true_negative_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    tp_pred_indexes = tf.where(tf.logical_and(tf.less(y_pred_label, threshold), tf.equal(y_true_label, 0)))
    return tf.reduce_mean(tf.gather(y_pred_label, tp_pred_indexes))

@tf.function
def mean_false_negative_value(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = tf.gather(y_true, [label_index], axis=1)
    y_pred_label = tf.gather(y_pred, [label_index], axis=1)
    y_true_label = y_true_label[:, 0]
    y_pred_label = y_pred_label[:, 0]
    tp_pred_indexes = tf.where(tf.logical_and(tf.less(y_pred_label, threshold), tf.equal(y_true_label, 1)))
    return tf.reduce_mean(tf.gather(y_pred_label, tp_pred_indexes))

@tf.function
def confusion_matrix_values_tf(y_true, y_pred, label_index=0, threshold=0.5):
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
        return [[np.nan, np.nan], [np.nan, np.nan]], accuracy, error_rate, np.nan, np.nan, np.nan, np.nan
    else:
        tn = predicted_negatives_in_true_set / (predicted_negatives + tf.keras.backend.epsilon())
        fp = (predicted_positives - predicted_positives_in_true_set) / (predicted_positives + K.epsilon())
        fn = (predicted_negatives - predicted_negatives_in_true_set) / (predicted_negatives + K.epsilon())
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return [[tp, fp], [fn, tn]], accuracy, error_rate, precision, recall, f1_score, np.nan


def confusion_matrix_values(y_true, y_pred, label_index=0, threshold=0.5):
    y_true_label = np.transpose(y_true)[label_index]
    y_pred_label = np.transpose(y_pred)[label_index]
    real_positive_indexes_mask = y_true_label > 0
    predicted_positive_indexes_mask = y_pred_label > threshold
    positive_indexes_intersect_mask = np.logical_and(predicted_positive_indexes_mask, real_positive_indexes_mask)
    predicted_positives = len(np.argwhere(predicted_positive_indexes_mask).flatten())
    predicted_positives_in_true_set = len(np.argwhere(positive_indexes_intersect_mask).flatten())
    tp = predicted_positives_in_true_set / (predicted_positives + tf.keras.backend.epsilon())
    real_negative_indexes_mask = y_true_label == 0
    predicted_negative_indexes_mask = y_pred_label < threshold
    negative_indexes_intersect_mask = np.logical_and(predicted_negative_indexes_mask, real_negative_indexes_mask)
    predicted_negatives = len(np.argwhere(predicted_negative_indexes_mask).flatten())
    predicted_negatives_in_true_set = len(np.argwhere(negative_indexes_intersect_mask).flatten())
    correct_predictions = predicted_positives_in_true_set + predicted_negatives_in_true_set
    total_predictions = len(y_true)
    accuracy = correct_predictions / (total_predictions + K.epsilon())
    error_rate = 1 - accuracy
    #TODO decide what values to be returned when no positives are detected
    if predicted_positives == 0:
        return [[np.nan, np.nan], [np.nan, np.nan]], accuracy, error_rate, np.nan, np.nan, np.nan
    else:
        tn = predicted_negatives_in_true_set / (predicted_negatives + K.epsilon())
        fp = (predicted_positives - predicted_positives_in_true_set) / (predicted_positives + K.epsilon())
        fn = (predicted_negatives - predicted_negatives_in_true_set) / (predicted_negatives + K.epsilon())
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return [[tp, fp], [fn, tn]], accuracy, error_rate, precision, recall, f1_score

def area_under_curve(y_true, y_pred, label_index=0, thresholds=100, method='roc'):
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
    nan_values = np.logical_and(np.isnan(recalls), np.isnan(precisions))
    x = recalls[~nan_values]
    y = precisions[~nan_values]
    threshold_values = threshold_values[~nan_values]
    return x, y, threshold_values, np.trapz(y, x)


class CategoricalClassMetric(tf.keras.metrics.Metric):
    def __init__(self, class_index, metric_function, batch_size, name="categorical_class_metric", metric_threshold=0.5,
                 **kwargs):
        super(CategoricalClassMetric, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.metric_function = metric_function
        self.metric_threshold = metric_threshold
        self.class_index = class_index
        self.metric = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.metric = self.metric_function(y_true, y_pred, self.class_index, self.metric_threshold)

    def result(self):
        return self.metric
