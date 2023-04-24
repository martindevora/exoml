import collections
import csv
import tensorflow as tf
import keras.backend as K
from keras.callbacks import CSVLogger
import numpy as np


class BatchAwareCsvLogger(CSVLogger):

    def __init__(self, filename, steps_per_epoch, separator=',', append=False):
        super().__init__(filename, separator, append)
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.__store_log(logs, self.steps_per_epoch - 1, self.current_epoch, False)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self.__store_log(logs, batch, self.current_epoch)

    def __store_log(self, logs, batch, epoch, with_validation=True):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        keys = logs.keys()
        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in keys)
        validation_fields = ['val_' + key for key in keys] if with_validation else []
        metric_fieldnames = list(keys) + validation_fields
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch', 'batch', 'lr'] + metric_fieldnames
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        learning_rate = K.eval(self.model.optimizer.lr)
        row_dict = {'epoch': epoch, 'batch': batch, 'lr': learning_rate}
        for metric_key in metric_fieldnames:
            row_dict[metric_key] = handle_value(logs[metric_key][0]
                                                if isinstance(logs[metric_key], (list, np.ndarray))
                                                else logs[metric_key]) if metric_key in logs else np.nan
        self.writer.writerow(row_dict)
        self.csv_file.flush()
