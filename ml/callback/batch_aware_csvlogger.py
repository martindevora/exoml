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
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
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
        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch', 'batch', 'learning_rate'] + self.keys
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        learning_rate = K.eval(self.model.optimizer._decayed_lr(tf.float64))
        row_dict = collections.OrderedDict({'epoch': self.current_epoch, 'batch': batch,
                                            'learning_rate': learning_rate})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
