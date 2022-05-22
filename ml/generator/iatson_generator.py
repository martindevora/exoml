import logging

import foldedleastsquares
from numpy.random import default_rng

from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd


class WatsonModelGenerator(tf.keras.utils.Sequence):
    TYPE_TO_LABEL = {'bckEB': [1, 0, 0], 'EB': [0, 1, 0], 'planet': [0, 0, 1], 'none': [0, 0, 0]}

    def __init__(self, injected_objects_file, batch_size, input_size, zero_epsilon, min_period=0.5, max_period=20):
        self.injected_objects_file = injected_objects_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.zero_epsilon = zero_epsilon
        self.injected_objects_df = pd.read_csv(injected_objects_file)
        self.random_number_generator = default_rng()
        self.min_period = min_period
        self.max_period = max_period

    def __len__(self):
        return (np.ceil(len(self.injected_objects_df) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        target_indexes = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size, 1)
        target_indexes = shuffle(target_indexes)
        injected_objects_df = self.injected_objects_df.iloc[target_indexes]
        #TODO determine batch_data_array size (multiple inputs)
        batch_data_array = np.empty(len(target_indexes), self.input_size)
        batch_data_values = np.empty((len(target_indexes), 3))
        i = 0
        for df_index, target_row in injected_objects_df.iterrows():
            target_id = target_row['TIC ID']
            lc_filename = str(target_id) + '_lc.csv'
            star_filename = str(target_id) + '_star.csv'
            star_df = pd.read_csv(star_filename)
            lc_df = pd.read_csv(lc_filename, usecols=['#time', 'flux', 'flux_err', 'centroid_x', 'centroid_y',
                                                      'motion_x', 'motion_y', 'bck_flux'], low_memory=True)
            type = target_row['type']
            batch_data_values[i] = self.TYPE_TO_LABEL[type]
            if type == 'none':
                #TODO determine a period far to collision with existing periods
                period = self.random_number_generator.uniform(self.min_period, self.max_period)
                min_lc_time = lc_df['#time'].min()
                epoch = self.random_number_generator.uniform(min_lc_time, min_lc_time + period)
            else:
                period = self.random_number_generator.uniform(self.min_period, self.max_period)
                min_lc_time = lc_df['#time'].min()
                epoch = self.random_number_generator.uniform(min_lc_time, min_lc_time + period)
            time = target_row['#time']
            target_row['#time_folded'] = foldedleastsquares.fold(time, period, epoch)
            target_row_sorted_fold = target_row.sort_values(by=['#time_folded'])
            #TODO append a quarter of the initial time to the end and shift to quarter
            # target_row_sorted_fold.iloc[0:len(target_row_sorted_fold) // 4]
            #['ld_a', 'ld_b', 'Teff', 'lum', 'logg', 'radius', 'mass', 'v', 'j', 'h', 'k'])
            batch_data_array[i][0] = star_df.iloc[0].to_numpy()
            batch_data_array[i][1] = target_row_sorted_fold.to_numpy()
            assert not np.isnan(batch_data_array[i]).any() and not np.isinf(batch_data_array[i]).any()
            values_df['model'] = 1 - ((1 - values_df['eb_model']) + (1 - values_df['bckeb_model']) + (1 - values_df['planet_model']))
            batch_data_values[i] = values_df['model'].to_numpy()
            batch_data_values[i] = np.nan_to_num(batch_data_values[i], nan=self.zero_epsilon)
            negative_values_args = np.argwhere(batch_data_values[i] <= 0).flatten()
            batch_data_values[i][negative_values_args] = self.zero_epsilon
            assert not np.isnan(batch_data_values[i]).any() and not np.isinf(batch_data_values[i]).any()
            logging.info("GENERATOR Inputs max " + str(np.max(batch_data_array[i])) + " and min " + str(np.min(batch_data_array[i])))
            logging.info("GENERATOR Values max " + str(np.max(batch_data_values[i])) + " and min " + str(np.min(batch_data_values[i])))
            i = i + 1
        return [batch_data_array], batch_data_values

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