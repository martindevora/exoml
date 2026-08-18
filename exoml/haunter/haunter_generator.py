import logging

from exoml.ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd


class HaunterModelGenerator(tf.keras.utils.Sequence):
    """
    Sequence generator for DETREND model batches
    """
    def __init__(self, dataset, batch_size, input_size, zero_epsilon=1e-7, shuffle=True):
        self.zero_epsilon = zero_epsilon
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        leng = (np.ceil(len(self.dataset) / float(self.batch_size))).astype(int)
        return leng

    def __getitem__(self, idx):
        batch_rows = self.dataset[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_data_array = batch_rows.loc[:, ['input_mass_Mearth_norm', 'input_CMF_norm', 'input_Zenv_norm',
                                              'input_Zwater_core_norm', 'input_Tsurf_K_norm', 'input_Psurf_bar_norm']]
        batch_data_values = batch_rows.loc[:, ['output_radius_Rearth_norm', 'output_entropy_log10_1000bar_SI_norm',
                                               'output_log10_f_s_SI_norm', 'output_log10_temperature_1000bar_K_norm']]
        #assert np.all((batch_data_array > 0) & (batch_data_array < 1))
        #assert np.all((batch_data_values > 0) & (batch_data_values < 1))
        return batch_data_array.to_numpy(dtype=np.float32), batch_data_values.to_numpy(dtype=np.float32)

    def class_weights(self):
        return None

    def get_dataset(self):
        x_spec = (
            tf.TensorSpec(shape=(None, 6, 1), dtype=tf.float32)
        )

        y_spec = tf.TensorSpec(shape=(None, 4,), dtype=tf.float32)  # Adjust if label shape is different

        return tf.data.Dataset.from_generator(
            lambda: iter(self),
            output_signature=(x_spec, y_spec)
        ).take(len(self)).repeat()
