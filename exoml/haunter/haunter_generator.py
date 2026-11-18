import numpy as np
import tensorflow as tf


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
        return batch_data_array.to_numpy(dtype=np.float32), batch_data_values.to_numpy(dtype=np.float32)

    def class_weights(self):
        return None
