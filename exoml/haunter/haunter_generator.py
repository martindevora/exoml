import numpy as np
import tensorflow as tf

# the inputs of the interior models as they are fed to the network: log10 of the mass and the surface pressure, and
# log10 of the distance of the surface pressure to the 1000 bar level where the entropy is measured (the entropy
# falls with the logarithm of that distance when the surface approaches the level)
INPUT_FEATURES = ['input_log10_mass_Mearth', 'input_CMF', 'input_Zenv', 'input_Zwater_core', 'input_Tsurf_K',
                  'input_log10_Psurf_bar', 'input_log10_distance_to_1000bar']
# the four outputs, all in the space they are learned in (log10 for the ones that span orders of magnitude)
OUTPUT_TARGETS = ['output_log10_radius_Rearth', 'output_entropy_log10_1000bar_SI', 'output_log10_f_s_SI',
                  'output_log10_temperature_1000bar_K']
NORM = '_norm'


class HaunterModelGenerator(tf.keras.utils.Sequence):
    """
    Sequence generator for Haunter batches
    """
    def __init__(self, dataset, batch_size, input_size, zero_epsilon=1e-7, shuffle=True):
        self.zero_epsilon = zero_epsilon
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        if self.shuffle:
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        self.inputs = self.dataset.loc[:, [c + NORM for c in INPUT_FEATURES]].to_numpy(dtype=np.float32)
        self.targets = self.dataset.loc[:, [c + NORM for c in OUTPUT_TARGETS]].to_numpy(dtype=np.float32)

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        return self.inputs[batch], self.targets[batch]

    def on_epoch_end(self):
        if self.shuffle:
            order = np.random.permutation(len(self.inputs))
            self.inputs, self.targets = self.inputs[order], self.targets[order]

    def class_weights(self):
        return None
