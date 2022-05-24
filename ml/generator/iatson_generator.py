import logging
import pathlib

import foldedleastsquares
import scipy
import wotan
from numpy.random import default_rng

from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd


class IatsonModelGenerator(tf.keras.utils.Sequence):
    TYPE_TO_LABEL = {'bckEB': [1, 0, 0, 0], 'EB': [0, 1, 0, 0], 'planet': [0, 0, 1, 0], 'none': [0, 0, 0, 1]}

    def __init__(self, injected_objects_df, lcs_dir, batch_size, input_sizes, zero_epsilon=1e-7,
                 min_period=0.5, max_period=20):
        self.injected_objects_df = injected_objects_df
        self.lcs_dir = lcs_dir
        self.batch_size = batch_size
        self.input_sizes = input_sizes
        self.zero_epsilon = zero_epsilon
        self.random_number_generator = default_rng()
        self.min_period = min_period
        self.max_period = max_period

    def __len__(self):
        return (np.ceil(len(self.injected_objects_df) / float(self.batch_size))).astype(int)

    def bin_interp(self, array, new_size):
        result = np.array(len(array))
        for index, nested_array in enumerate(array):
            arr_interp = scipy.interpolate.interp1d(np.arange(array.size), array)
            result[index] = arr_interp(np.linspace(0, array.size - 1, new_size))
        return result

    def __getitem__(self, idx):
        target_indexes = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size, 1)
        target_indexes = shuffle(target_indexes)
        injected_objects_df = self.injected_objects_df.iloc[target_indexes]
        #TODO determine batch_data_array size (multiple inputs)
        batch_data_array = np.empty((len(target_indexes), len(self.input_sizes)))
        batch_data_values = np.empty((len(target_indexes), 3))
        i = 0
        for df_index, target_row in injected_objects_df.iterrows():
            target_id = int(target_row['TIC ID'])
            leading_zeros_object_id = '{:09}'.format(target_id)
            lc_filename = str(leading_zeros_object_id) + '_lc.csv'
            star_filename = str(leading_zeros_object_id) + '_star.csv'
            star_df = pd.read_csv(self.lcs_dir + '/' + star_filename)
            lc_df = pd.read_csv(self.lcs_dir + '/' + lc_filename, usecols=['#time', 'flux', 'flux_err', 'centroid_x',
                                                                           'centroid_y', 'motion_x', 'motion_y', 'bck_flux'], low_memory=True)
            type = target_row['type']
            batch_data_values[i] = self.TYPE_TO_LABEL[type]
            not_null_times_args = np.argwhere(lc_df['#time'].to_numpy() > 0).flatten()
            lc_df = lc_df.iloc[not_null_times_args]
            if type == 'none':
                #TODO determine a period far to collision with existing periods
                period = self.random_number_generator.uniform(self.min_period, self.max_period)
                min_lc_time = lc_df['#time'].min()
                epoch = self.random_number_generator.uniform(min_lc_time, min_lc_time + period)
                duration = wotan.t14(star_df['radius'], star_df['mass'], period)
            else:
                period = target_row['period']
                epoch = target_row['epoch']
                duration = target_row['duration(h)'] / 24
            duration_to_period = duration / period
            #['ld_a', 'ld_b', 'Teff', 'lum', 'logg', 'radius', 'mass', 'v', 'j', 'h', 'k'])
            batch_data_array[i][0] = star_df.to_numpy()
            time = lc_df['#time']
            #TODO we need to bin the data to match input sizes yet
            # Global flux
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period, epoch)
            lc_df_sorted_fold = lc_df.sort_values(by=['#time_folded'])
            # Shifting data 1/4 so that main transit and possible occultation don't get cut by the borders
            lc_df_sorted_fold['#time_folded'] = lc_df_sorted_fold['#time_folded'] + 0.25
            lc_df_sorted_fold = lc_df_sorted_fold.sort_values(by=['#time_folded'])
            lc_df_sorted_fold.drop('#time', inplace=True, axis=1)
            batch_data_array[i][1] = self.bin_interp(lc_df_sorted_fold.to_numpy(), self.input_sizes[1])
            # Focus flux even
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period, epoch)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][2] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[2])
            # Focus flux odd
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period, epoch + period / 2)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][3] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[3])
            # Focus flux sub-harmonic even
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period / 2, epoch)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][4] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[4])
            # Focus flux sub-harmonic odd
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period / 2, epoch + period / 4)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][5] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[5])
            # Focus flux harmonic even
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period * 2, epoch)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][6] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[6])
            # Focus flux harmonic odd
            lc_df['#time_folded'] = foldedleastsquares.fold(time, period * 2, epoch + period * 3 / 2)
            lc_df_focus = lc_df.sort_values(by=['#time_folded'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time_folded'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time_folded'] < 0.5 + duration_to_period * 3)]
            lc_df_focus.drop('#time', inplace=True, axis=1)
            batch_data_array[i][7] = self.bin_interp(lc_df_focus.to_numpy(), self.input_sizes[7])
            assert not np.isnan(batch_data_array[i]).any() and not np.isinf(batch_data_array[i]).any()
            logging.info("GENERATOR Inputs max " + str(np.max(batch_data_array[i])) + " and min " + str(np.min(batch_data_array[i])))
            i = i + 1
        return batch_data_array, batch_data_values

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

df = pd.read_csv("/mnt/DATA-2/ete6/injected_objects.csv")
df = shuffle(df)
df.reset_index(inplace=True, drop=True)
img = IatsonModelGenerator(df, "/mnt/DATA-2/ete6/lcs", 20, [11, 2500, 500, 500, 500, 500, 500, 500])
img.__getitem__(20)
