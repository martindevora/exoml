import logging

from keras.callbacks import EarlyStopping


class ExoMlEarlyStopping(EarlyStopping):
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        if self.stopped_epoch > 0 and self.verbose > 0:
            logging.warning('Epoch %05d: early stopping', (self.stopped_epoch + 1))
