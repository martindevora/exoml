import matplotlib.pyplot as plt
from numpy.random import default_rng
from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd

from ml.physics.transit_functions import t14


class TranspotModelGenerator(tf.keras.utils.Sequence):
    TYPE_TO_LABEL = {'bckEB': [1, 0, 0, 0], 'EB': [0, 1, 0, 0], 'planet': [0, 0, 1, 0], 'none': [0, 0, 0, 1]}

    def __init__(self, injected_objects_df, lcs_dir, batch_size, input_sizes, zero_epsilon=1e-7,
                 min_period=0.5, max_period=10, measurements_per_point=13):
        self.injected_objects_df = injected_objects_df
        self.lcs_dir = lcs_dir
        self.batch_size = batch_size
        self.input_sizes = input_sizes
        self.zero_epsilon = zero_epsilon
        self.random_number_generator = default_rng()
        self.min_period = min_period
        self.max_period = max_period
        self.measurements_per_point = measurements_per_point

    def __len__(self):
        return (np.ceil(len(self.injected_objects_df) / float(self.batch_size))).astype(int)

    def _prepare_input_lc(self, input_df):
        time = input_df["#time"].to_numpy()
        # We don't want time encoding because time will be used for folding and then dropped
        input_df["#time"] = value_encode_times(time - time[0])
        dif = time[1:] - time[:-1]
        jumps = np.where(np.abs(dif) > 0.25)[0]
        jumps = np.append(jumps, len(input_df))
        previous_jump_index = 0
        for jumpIndex in jumps:
            token = input_df["centroid_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_x"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["centroid_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_y"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["motion_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_x"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["motion_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_y"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["bck_flux"][previous_jump_index:jumpIndex].to_numpy()
            input_df["bck_flux"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 3
            previous_jump_index = jumpIndex
        input_df = input_df.fillna(self.zero_epsilon)
        input_df = input_df.replace(0.0, self.zero_epsilon)
        input_df = input_df.replace(0, self.zero_epsilon)
        return input_df

    def _prepare_input_star(self, star_df):
        star_df['ld_a'] = star_df['ld_a'].fillna(0.25)
        star_df['ld_b'] = star_df['ld_b'].fillna(0.25)
        star_df['Teff'] = star_df['Teff'].fillna(6000)
        star_df['radius'] = star_df['radius'].fillna(1)
        star_df['mass'] = star_df['mass'].fillna(1)
        star_df['lum'] = star_df['lum'].fillna(1)
        star_df['logg'] = star_df['logg'].fillna(1)
        star_df['v'] = star_df['v'].fillna(-10)
        star_df['j'] = star_df['j'].fillna(-10)
        star_df['k'] = star_df['k'].fillna(-10)
        star_df['h'] = star_df['h'].fillna(-10)
        star_df['Teff'] = star_df['Teff'] / 40000
        star_df['radius'] = star_df['radius'] / 3
        star_df['radius'] = star_df['radius'] if star_df['radius'].iloc[0] <= 1 else 1
        star_df['mass'] = star_df['mass'] / 3
        star_df['mass'] = star_df['mass'] if star_df['mass'].iloc[0] <= 1 else 1
        star_df['lum'] = star_df['mass'] / 1000
        star_df['logg'] = star_df['logg'] / 100
        star_df['v'] = (7 - star_df['v']) / 27
        star_df['j'] = (7 - star_df['j']) / 27
        star_df['k'] = (7 - star_df['k']) / 27
        star_df['h'] = (7 - star_df['h']) / 27
        return star_df

    def _plot_df(self, df, type, scenario):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        axs[0][0].scatter(df['#time'], df['flux'])
        axs[0][1].scatter(df['#time'], df['centroid_x'])
        axs[0][2].scatter(df['#time'], df['centroid_y'])
        axs[1][0].scatter(df['#time'], df['bck_flux'])
        axs[1][1].scatter(df['#time'], df['motion_x'])
        axs[1][2].scatter(df['#time'], df['motion_y'])
        plt.title(type + " " + scenario)
        plt.show()
        plt.clf()
        plt.close()

    def _plot_input(self, input_array, type, scenario):
        fig, axs = plt.subplots(4, 3, figsize=(12, 6), constrained_layout=True)
        axs[0][0].scatter(np.transpose(input_array)[0], np.transpose(input_array)[1])
        axs[0][1].scatter(np.transpose(input_array)[0], np.transpose(input_array)[3])
        axs[0][2].scatter(np.transpose(input_array)[0], np.transpose(input_array)[4])
        axs[1][0].scatter(np.transpose(input_array)[0], np.transpose(input_array)[5])
        axs[1][1].scatter(np.transpose(input_array)[0], np.transpose(input_array)[6])
        axs[1][2].scatter(np.transpose(input_array)[0], np.transpose(input_array)[7])
        axs[2][0].scatter(np.transpose(input_array)[0], np.transpose(input_array)[8])
        axs[2][1].scatter(np.transpose(input_array)[0], np.transpose(input_array)[9])
        axs[2][2].scatter(np.transpose(input_array)[0], np.transpose(input_array)[10])
        axs[3][0].scatter(np.transpose(input_array)[0], np.transpose(input_array)[11])
        axs[3][1].scatter(np.transpose(input_array)[0], np.transpose(input_array)[12])
        plt.title(type + " " + scenario)
        plt.show()
        plt.clf()
        plt.close()

    def __getitem__(self, idx):
        max_index = (idx + 1) * self.batch_size
        max_index = len(self.injected_objects_df) if max_index > len(self.injected_objects_df) else max_index
        target_indexes = np.arange(idx * self.batch_size, max_index, 1)
        target_indexes = shuffle(target_indexes)
        injected_objects_df = self.injected_objects_df.iloc[target_indexes]
        star_array = np.empty((len(target_indexes), 11, 1))
        global_flux_array = np.empty((len(target_indexes), self.input_sizes[0], self.measurements_per_point))
        batch_data_values = np.empty((len(target_indexes), 1))
        i = 0
        for df_index, target_row in injected_objects_df.iterrows():
            target_id = int(target_row['TIC ID'])
            leading_zeros_object_id = '{:09}'.format(target_id)
            lc_filename = str(leading_zeros_object_id) + '_lc.csv'
            star_filename = str(leading_zeros_object_id) + '_star.csv'
            star_df = pd.read_csv(self.lcs_dir + '/' + star_filename, index_col=False)
            star_df = self._prepare_input_star(star_df)
            lc_df = pd.read_csv(self.lcs_dir + '/' + lc_filename, usecols=['#time', 'flux', 'flux_0', 'flux_1',
                                                                           'flux_2', 'flux_3', 'flux_4',
                                                                           'flux_err', 'centroid_x',
                                                                           'centroid_y', 'motion_x', 'motion_y',
                                                                           'bck_flux'], low_memory=True)
            lc_df = self._prepare_input_lc(lc_df)
            type = target_row['type']
            batch_data_values[i] = 0 if type == 'none' else 1
            not_null_times_args = np.argwhere(lc_df['#time'].to_numpy() > 0).flatten()
            lc_df = lc_df.iloc[not_null_times_args]
            star_df.drop('Unnamed: 0', inplace=True, axis=1)
            star_array[i] = np.transpose(star_df.to_numpy())
            global_flux_array[i] = lc_df.to_numpy()
            assert not np.isnan(star_array[i]).any() and not np.isinf(star_array[i]).any()
            assert not np.isnan(global_flux_array[i]).any() and not np.isinf(global_flux_array[i]).any()
            assert np.max(global_flux_array[i]) < 2
            i = i + 1
        return [star_array, global_flux_array], batch_data_values

    def __prepare_input_data(self, input_df):
        time = input_df["#time"].to_numpy()
        input_df["#time"] = value_encode_times(time - time[0])
        dif = time[1:] - time[:-1]
        jumps = np.where(np.abs(dif) > 0.25)[0]
        jumps = np.append(jumps, len(input_df))
        previous_jump_index = 0
        for jumpIndex in jumps:
            token = input_df["centroid_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_x"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["centroid_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_y"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["motion_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_x"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["motion_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_y"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 2
            token = input_df["bck_flux"][previous_jump_index:jumpIndex].to_numpy()
            input_df["bck_flux"][previous_jump_index:jumpIndex] = (np.tanh(token - np.nanmedian(token)) + 1) / 3
            previous_jump_index = jumpIndex
        input_df = input_df.fillna(self.zero_epsilon)
        input_df = input_df.replace(0.0, self.zero_epsilon)
        input_df = input_df.replace(0, self.zero_epsilon)
        return input_df

# df = pd.read_csv("/mnt/DATA-2/ete6/injected_objects.csv", index_col=False)
# df = shuffle(df)
# df.reset_index(inplace=True, drop=True)
# img = IatsonModelGenerator(df, "/mnt/DATA-2/ete6/lcs", 20, [11, 2500, 500, 500, 500, 500, 500, 500])
# img.__getitem__(20)
