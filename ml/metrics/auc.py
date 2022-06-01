import collections
import csv

import pandas
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from ml.callback.training_data_aware import ValidationDataAwareCallback


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
        tn = predicted_negatives_in_true_set / (predicted_negatives + tf.keras.backend.epsilon())
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


class ConfusionMatrixStatisticsCallback(ValidationDataAwareCallback):
    def __init__(self, filenames, validation_data=None):
        super().__init__(validation_data=validation_data)
        self.filenames = filenames
        self.stats_csv_files = []
        self.writers = []
        self.roc_dfs = []
        self.keys = None
        self.roc_df = pandas.DataFrame(columns=['epoch', 'label', 'precision', 'recall', 'thresholds'])

    def on_train_begin(self, logs=None):
        mode = 'w'
        for filename in self.filenames:
            self.stats_csv_files = self.stats_csv_files + [tf.io.gfile.GFile(self.model_dir + '/' + filename +
                                                                             '_stats.csv', mode)]

    def on_train_end(self, logs=None):
        for index, filename in enumerate(self.filenames):
            self.stats_csv_files[index].close()
            self.writers[index] = None
        self.writers = None
        self.roc_dfs = None

    def on_epoch_end(self, epoch, logs={}):
        # x_train, y_train = self.train[0], self.train[1]
        # y_pred = self.model.predict(x_train)
        # confusion_matrix, accuracy, error_rate, precision, recall, f1_score = self.confusion_matrix(y_train, y_pred)
        # logs['F1_score_train'] = np.round(f1_score, 5)
        # if self.validation:
        batches = len(self.validation_data)
        y_valid = None
        for batch in range(batches):
            x_val, y_val = self.validation_data[batch]
            if y_valid is None:
                y_valid = y_val
            else:
                y_valid = np.concatenate((y_valid, y_val))
        y_val_pred = self.model.predict(self.validation_data)
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())
        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)
        if len(self.writers) == 0:
            class CustomDialect(csv.excel):
                delimiter = ','
            fieldnames = ['epoch', 'val_f1_score', 'val_tp', 'val_fp', 'val_fn', 'val_tn', 'val_precision',
                          'val_recall', 'val_roc'] + self.keys
            for csv_file in self.stats_csv_files:
                new_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, dialect=CustomDialect)
                new_writer.writeheader()
                self.writers = self.writers + [new_writer]

        for index, filename in enumerate(self.filenames):
            confusion_matrix, accuracy, error_rate, precision, recall, f1_score = \
                confusion_matrix_values(y_valid, y_val_pred, label_index=index)
            recalls, precisions, thresholds, roc = area_under_curve(y_valid, y_val_pred, label_index=index)
            row_dict = collections.OrderedDict({'epoch': epoch, 'val_f1_score': f1_score,
                                                'val_tp': confusion_matrix[0][0],
                                                'val_fp': confusion_matrix[0][1],
                                                'val_fn': confusion_matrix[1][0],
                                                'val_tn': confusion_matrix[1][1],
                                                'val_precision': precision,
                                                'val_recall': recall,
                                                'val_roc': roc
                                                })
            self.writers[index].writerow(row_dict)
            self.stats_csv_files[index].flush()
            for precision_index, precision in enumerate(precisions):
                self.roc_df = self.roc_df.append({'epoch': epoch, 'label': index, 'recall': recalls[precision_index],
                                                  'precision': precisions[precision_index],
                                                  'thresholds': thresholds[precision_index]}, ignore_index=True)
        self.roc_df.to_csv(self.model_dir + '/precision_recall.csv', index=False)
