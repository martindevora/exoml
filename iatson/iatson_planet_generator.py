import logging
import os
import sys

import matplotlib.pyplot as plt
import foldedleastsquares
from numpy.random import default_rng
from ete6.ete6_generator import Ete6ModelGenerator
from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import numpy as np
import pandas as pd


class IatsonPlanetModelGenerator(Ete6ModelGenerator):
    def __init__(self, injected_objects_df, lcs_dir, batch_size, input_sizes, type_to_label, zero_epsilon=1e-7,
                 measurements_per_point=12, plot_inputs=False, fixed_target_id=None, store_arrays=False,
                 from_arrays=True):
        super().__init__(zero_epsilon)
        self.injected_objects_df = injected_objects_df
        self.lcs_dir = lcs_dir
        self.batch_size = batch_size
        self.input_sizes = input_sizes
        self.random_number_generator = default_rng()
        self.measurements_per_point = measurements_per_point
        self.type_to_label = type_to_label
        self.plot_inputs = plot_inputs
        self.fixed_target_id = fixed_target_id
        self.store_arrays = store_arrays
        self.from_arrays = from_arrays

    def __len__(self):
        return (np.ceil(len(self.injected_objects_df) / float(self.batch_size))).astype(int)

    def class_weights(self):
        return {0: 1, 1: 1}

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

    def _plot_input(self, input_array, input_err_array, type, scenario):
        if self.plot_inputs:
            transposed_err_array = np.transpose(input_err_array)
            transposed_array = np.transpose(input_array)
            fig, axs = plt.subplots(6, 2, figsize=(24, 12), constrained_layout=True)
            current_array = transposed_array[0]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            time_array = current_array
            axs[0][0].scatter(np.arange(0, len(current_array)), current_array)
            axs[0][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[0][0].set_title("Time")
            current_array = transposed_array[1]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[0][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[1][current_array_mask], color="orange", alpha=0.5)
            axs[0][1].scatter(time_array[current_array_mask], current_array)
            axs[0][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[0][1].set_title("Flux")
            current_array = transposed_array[7]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[1][0].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[7][current_array_mask], color="orange", alpha=0.5)
            axs[1][0].scatter(time_array[current_array_mask], current_array)
            axs[1][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array),
                               np.mean(current_array) + 6 * np.std(current_array))
            axs[1][0].set_title("Flux 1")
            current_array = transposed_array[8]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[1][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[8][current_array_mask], color="orange", alpha=0.5)
            axs[1][1].scatter(time_array[current_array_mask], current_array)
            axs[1][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array),
                               np.mean(current_array) + 6 * np.std(current_array))
            axs[1][1].set_title("Flux 2")
            current_array = transposed_array[9]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[2][0].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[9][current_array_mask], color="orange", alpha=0.5)
            axs[2][0].scatter(time_array[current_array_mask], current_array)
            axs[2][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array),
                               np.mean(current_array) + 6 * np.std(current_array))
            axs[2][0].set_title("Flux 3")
            current_array = transposed_array[10]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[2][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[10][current_array_mask], color="orange", alpha=0.5)
            axs[2][1].scatter(time_array[current_array_mask], current_array)
            axs[2][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array),
                               np.mean(current_array) + 6 * np.std(current_array))
            axs[2][1].set_title("Flux 4")
            current_array = transposed_array[11]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[3][0].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[11][current_array_mask], color="orange", alpha=0.5)
            axs[3][0].scatter(time_array[current_array_mask], current_array)
            axs[3][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array),
                               np.mean(current_array) + 6 * np.std(current_array))
            axs[3][0].set_title("Flux 5")
            current_array = transposed_array[6]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[3][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[6][current_array_mask], color="orange", alpha=0.5)
            axs[3][1].scatter(time_array[current_array_mask], current_array)
            axs[3][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[3][1].set_title("Bck Flux")
            current_array = transposed_array[2]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[4][0].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[2][current_array_mask], color="orange", alpha=0.5)
            axs[4][0].scatter(time_array[current_array_mask], current_array)
            axs[4][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[4][0].set_title("Centroid X")
            current_array = transposed_array[3]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[4][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[3][current_array_mask], color="orange", alpha=0.5)
            axs[4][1].scatter(time_array[current_array_mask], current_array)
            axs[4][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[4][1].set_title("Centroid Y")
            current_array = transposed_array[4]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[5][0].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[4][current_array_mask], color="orange", alpha=0.5)
            axs[5][0].scatter(time_array[current_array_mask], current_array)
            axs[5][0].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[5][0].set_title("Motion Y")
            current_array = transposed_array[5]
            current_array_mask = np.argwhere((~np.isnan(current_array)) & (current_array > self.zero_epsilon)).flatten()
            current_array = current_array[current_array_mask]
            axs[5][1].errorbar(time_array[current_array_mask], current_array,
                               yerr=transposed_err_array[5][current_array_mask], color="orange", alpha=0.5)
            axs[5][1].scatter(time_array[current_array_mask], current_array)
            axs[5][1].set_ylim(np.mean(current_array) - 6 * np.std(current_array), np.mean(current_array) + 6 * np.std(current_array))
            axs[5][1].set_title("Motion Y")
            fig.suptitle(type + " " + scenario)
            plt.show()
            plt.clf()
            plt.close()

    def plot_single_data(self, lc_df, target_row):
        fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        axs.scatter(lc_df['#time'], lc_df['flux_2'])
        axs.set_ylim(0.5 - target_row['tce_depth'] / 0.5e6, 0.505)
        plt.show()
        plt.clf()
        plt.close()

    def __getitem__(self, idx):
        max_index = (idx + 1) * self.batch_size
        max_index = len(self.injected_objects_df) if max_index > len(self.injected_objects_df) else max_index
        target_indexes = np.arange(idx * self.batch_size, max_index, 1)
        injected_objects_df = self.injected_objects_df.iloc[target_indexes]
        star_array = np.empty((len(target_indexes), self.input_sizes[0], 1))
        star_neighbours_array = np.empty((len(target_indexes), self.input_sizes[1], 1))
        global_flux_array = np.empty((len(target_indexes), self.input_sizes[2], self.measurements_per_point))
        folded_flux_even_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_even_subhar_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_subhar_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_even_har_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_har_array = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        global_flux_array_err = np.empty((len(target_indexes), self.input_sizes[2], self.measurements_per_point))
        folded_flux_even_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_even_subhar_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_subhar_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_even_har_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        folded_flux_odd_har_array_err = np.empty((len(target_indexes), self.input_sizes[3], self.measurements_per_point))
        batch_data_values = np.empty((len(target_indexes), 1))
        i = 0
        for df_index, target_row in injected_objects_df.iterrows():
            if self.fixed_target_id is not None:
                target_row = self.injected_objects_df[self.injected_objects_df['object_id'] == self.fixed_target_id].iloc[0]
            #TODO refactor to use mission for object_id
            object_id = target_row['object_id'].split(' ')
            mission_id = object_id[0]
            target_id = int(object_id[1])
            period = target_row['period']
            type = target_row['type']
            batch_data_values[i] = self.type_to_label[type]
            file_prefix = self.lcs_dir + '/' + mission_id + '_' + str(target_id) + '_' + str(np.round(period, 2))
            if self.from_arrays:
                file_prefix = self.lcs_dir + '/' + mission_id + '_' + str(target_id)
                star_array[i] = np.loadtxt(file_prefix + '_input_star.csv', delimiter=',')
                star_neighbours_array[i] = np.loadtxt(file_prefix + '_input_nb.csv', delimiter=',')
                global_flux_array[i] = np.loadtxt(file_prefix + '_input_global.csv', delimiter=',')
                global_flux_array_err[i] = np.loadtxt(file_prefix + '_input_global_err.csv', delimiter=',')
                folded_flux_even_array[i] = np.loadtxt(file_prefix + '_input_even.csv', delimiter=',')
                folded_flux_even_array_err[i] = np.loadtxt(file_prefix + '_input_even_err.csv', delimiter=',')
                folded_flux_odd_array[i] = np.loadtxt(file_prefix + '_input_odd.csv', delimiter=',')
                folded_flux_odd_array_err[i] = np.loadtxt(file_prefix + '_input_odd_err.csv', delimiter=',')
                folded_flux_even_subhar_array[i] = np.loadtxt(file_prefix + '_input_even_sh.csv', delimiter=',')
                folded_flux_even_subhar_array_err[i] = np.loadtxt(file_prefix + '_input_even_sh_err.csv', delimiter=',')
                folded_flux_odd_subhar_array[i] = np.loadtxt(file_prefix + '_input_odd_sh.csv', delimiter=',')
                folded_flux_odd_subhar_array_err[i] = np.loadtxt(file_prefix + '_input_odd_sh_err.csv', delimiter=',')
                folded_flux_even_har_array[i] = np.loadtxt(file_prefix + '_input_even_h.csv', delimiter=',')
                folded_flux_even_har_array_err[i] = np.loadtxt(file_prefix + '_input_even_h_err.csv', delimiter=',')
                folded_flux_odd_har_array[i] = np.loadtxt(file_prefix + '_input_odd_h.csv', delimiter=',')
                folded_flux_odd_har_array_err[i] = np.loadtxt(file_prefix + '_input_odd_h_err.csv', delimiter=',')
            else:
                #TODO refactor to use mission for _lc files
                lc_filename = self.lcs_dir + '/' + mission_id + '_' + str(target_id) + '_lc.csv'
                lc_short_filename = self.lcs_dir + '/' + mission_id + '_' + str(target_id) + '_lc_short.csv'
                lc_long_filename = self.lcs_dir + '/' + mission_id + '_' + str(target_id) + '_lc_long.csv'
                lc_df = None
                if os.path.exists(lc_filename):
                    read_df = pd.read_csv(lc_filename, usecols=['#time', 'flux', 'flux_0', 'flux_1',
                                                                  'flux_2', 'flux_3', 'flux_4',
                                                                  'centroid_x', 'centroid_y', 'motion_x', 'motion_y',
                                                                  'bck_flux'], low_memory=True)
                    lc_df = read_df if lc_df is None else lc_df.append(read_df, ignore_index=True)
                if os.path.exists(lc_long_filename):
                    read_df = pd.read_csv(lc_long_filename, usecols=['#time', 'flux', 'flux_0', 'flux_1',
                                                                'flux_2', 'flux_3', 'flux_4',
                                                                'centroid_x', 'centroid_y', 'motion_x', 'motion_y',
                                                                'bck_flux'], low_memory=True)
                    lc_df = read_df if lc_df is None else lc_df.append(read_df, ignore_index=True)
                if os.path.exists(lc_short_filename):
                    read_df = pd.read_csv(lc_short_filename, usecols=['#time', 'flux', 'flux_0', 'flux_1',
                                                                'flux_2', 'flux_3', 'flux_4',
                                                                'centroid_x', 'centroid_y', 'motion_x', 'motion_y',
                                                                'bck_flux'], low_memory=True)
                    lc_df = read_df if lc_df is None else lc_df.append(read_df, ignore_index=True)
                if lc_df is None:
                    logging.warning("No curve for target " + str(target_row['object_id']))
                assert lc_df is not None
                lc_df = lc_df.sort_values(by=['#time'])
                #TODO refactor to use mission for star files
                star_filename = self.lcs_dir + '/' + mission_id + '_' + str(target_id) + '_star.csv'
                star_neighbours_df = pd.read_csv(star_filename,
                                                 usecols=['Teff', 'lum', 'v', 'j', 'k', 'h', 'radius', 'mass',
                                                          'dist_arcsec'], index_col=False)
                star_neighbours_df = self._prepare_input_neighbour_stars(star_neighbours_df)
                star_df = pd.read_csv(star_filename, index_col=False)
                star_df = self._prepare_input_star(star_df)
                lc_df = self._prepare_input_lc(lc_df)
                not_null_times_args = np.argwhere(lc_df['#time'].to_numpy() > 0).flatten()
                lc_df = lc_df.iloc[not_null_times_args]
                epoch = target_row['epoch']
                duration = target_row['duration(h)'] / 24
                duration_to_period = duration / period
                #['ld_a', 'ld_b', 'Teff', 'lum', 'logg', 'radius', 'mass', 'v', 'j', 'h', 'k'])
                neighbours_array = star_neighbours_df.to_numpy().flatten()
                star_neighbours_array[i] = np.transpose([neighbours_array if len(neighbours_array) == 9 * 15 \
                    else neighbours_array + np.zeros(9 * 15 - len(neighbours_array))])
                star_array[i] = np.transpose([star_df.to_numpy()])
                time = lc_df['#time'].to_numpy()
                # Global flux
                # Shifting data 1/4 so that main transit and possible occultation don't get cut by the borders
                lc_df_sorted_fold = lc_df.copy()
                lc_df_sorted_fold['#time'] = foldedleastsquares.fold(time, period, epoch + period / 4)
                lc_df_sorted_fold = lc_df_sorted_fold.sort_values(by=['#time'])
                global_flux_array[i], global_flux_array_err[i] = \
                    self.bin_by_time(lc_df_sorted_fold.to_numpy(), self.input_sizes[2], target_row['object_id'])
                self._plot_input(global_flux_array[i], global_flux_array_err[i], type, "global")
                # Focus flux even
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period, epoch)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 2) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 2)]
                folded_flux_even_array[i], folded_flux_even_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[3], target_row['object_id'])
                self._plot_input(folded_flux_even_array[i], folded_flux_even_array_err[i], type, "even")
                # Focus flux odd
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period, epoch + period / 2)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
                folded_flux_odd_array[i], folded_flux_odd_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[4], target_row['object_id'])
                self._plot_input(folded_flux_odd_array[i], folded_flux_odd_array_err[i], type, "odd")
                # Focus flux sub-harmonic even
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period / 2, epoch)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
                folded_flux_even_subhar_array[i], folded_flux_even_subhar_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[5], target_row['object_id'])
                self._plot_input(folded_flux_even_subhar_array[i], folded_flux_even_subhar_array_err[i], type, "even_subhar")
                # Focus flux sub-harmonic odd
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period / 2, epoch + period / 4)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
                folded_flux_odd_subhar_array[i], folded_flux_odd_subhar_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[6], target_row['object_id'])
                self._plot_input(folded_flux_odd_subhar_array[i], folded_flux_odd_subhar_array_err[i], type, "odd_subhar")
                # Focus flux harmonic even
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period * 2, epoch)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
                folded_flux_even_har_array[i], folded_flux_even_har_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[7], target_row['object_id'])
                self._plot_input(folded_flux_even_har_array[i], folded_flux_even_har_array[i], type, "even_har")
                # Focus flux harmonic odd
                lc_df_focus = lc_df.copy()
                lc_df_focus['#time'] = foldedleastsquares.fold(time, period * 2, epoch + period * 3 / 2)
                lc_df_focus = lc_df_focus.sort_values(by=['#time'])
                lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                          (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
                folded_flux_odd_har_array[i], folded_flux_odd_har_array_err[i] = \
                    self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[8], target_row['object_id'])
                self._plot_input(folded_flux_odd_har_array[i], folded_flux_odd_har_array_err[i], type, "odd_har")
                if np.max(global_flux_array[i]) >= 2:
                    logging.error("Wrong flux value for target " + str(target_row['object_id']))
                assert np.max(global_flux_array[i]) < 2
                assert not np.isnan(star_array[i]).any() and not np.isinf(star_array[i]).any()
                assert not np.isnan(global_flux_array[i]).any() and not np.isinf(global_flux_array[i]).any()
                assert not np.isnan(folded_flux_even_array[i]).any() and not np.isinf(folded_flux_even_array[i]).any()
                assert not np.isnan(folded_flux_odd_array[i]).any() and not np.isinf(folded_flux_odd_array[i]).any()
                assert not np.isnan(folded_flux_even_subhar_array[i]).any() and not np.isinf(folded_flux_even_subhar_array[i]).any()
                assert not np.isnan(folded_flux_odd_subhar_array[i]).any() and not np.isinf(folded_flux_odd_subhar_array[i]).any()
                assert not np.isnan(folded_flux_even_har_array[i]).any() and not np.isinf(folded_flux_even_har_array[i]).any()
                assert not np.isnan(folded_flux_odd_har_array[i]).any() and not np.isinf(folded_flux_odd_har_array[i]).any()
                # logging.info("GENERATOR Star Inputs max " + str(np.max(star_array[i])) + " and min " +
                #              str(np.min(star_array[i])))
                # logging.info("GENERATOR Global Flux Inputs max " + str(np.max(global_flux_array[i])) + " and min " +
                #              str(np.min(global_flux_array[i])))
                # logging.info("GENERATOR Folded Even Inputs max " + str(np.max(folded_flux_even_array[i])) + " and min " +
                #              str(np.min(folded_flux_even_array[i])))
                # logging.info("GENERATOR Folded Odd Inputs max " + str(np.max(folded_flux_odd_array[i])) + " and min " +
                #              str(np.min(folded_flux_odd_array[i])))
                # logging.info("GENERATOR Folded Even Subhar Inputs max " + str(np.max(folded_flux_even_subhar_array[i])) +
                #              " and min " + str(np.min(folded_flux_even_subhar_array[i])))
                # logging.info("GENERATOR Folded Odd Subhar Inputs max " + str(np.max(folded_flux_odd_subhar_array[i])) +
                #              " and min " + str(np.min(folded_flux_odd_subhar_array[i])))
                # logging.info("GENERATOR Folded Even Har Inputs max " + str(np.max(folded_flux_even_har_array[i])) +
                #              " and min " + str(np.min(folded_flux_even_har_array[i])))
                # logging.info("GENERATOR Folded Odd Har Inputs max " + str(np.max(folded_flux_odd_har_array[i])) +
                #              " and min " + str(np.min(folded_flux_odd_har_array[i])))
            if self.store_arrays:
                logging.info("Storing arrays into prefix " + file_prefix)
                np.savetxt(file_prefix + '_input_star.csv', star_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_nb.csv', star_neighbours_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_global.csv', global_flux_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_global_err.csv', global_flux_array_err[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_even.csv', folded_flux_even_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_even_err.csv', folded_flux_even_array_err[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_odd.csv', folded_flux_odd_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_odd_err.csv', folded_flux_odd_array_err[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_even_sh.csv', folded_flux_even_subhar_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_even_sh_err.csv', folded_flux_even_subhar_array_err[i], delimiter=',',
                           fmt='%f')
                np.savetxt(file_prefix + '_input_odd_sh.csv', folded_flux_odd_subhar_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_odd_sh_err.csv', folded_flux_odd_subhar_array_err[i], delimiter=',',
                           fmt='%f')
                np.savetxt(file_prefix + '_input_even_h.csv', folded_flux_even_har_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_even_h_err.csv', folded_flux_even_har_array_err[i], delimiter=',',
                           fmt='%f')
                np.savetxt(file_prefix + '_input_odd_h.csv', folded_flux_odd_har_array[i], delimiter=',', fmt='%f')
                np.savetxt(file_prefix + '_input_odd_h_err.csv', folded_flux_odd_har_array_err[i], delimiter=',', fmt='%f')
            i = i + 1
        return [star_array, star_neighbours_array, global_flux_array, global_flux_array_err, folded_flux_even_array,
                folded_flux_even_array_err, folded_flux_odd_array, folded_flux_odd_array_err,
                folded_flux_even_subhar_array, folded_flux_even_subhar_array_err, folded_flux_odd_subhar_array,
                folded_flux_odd_subhar_array_err, folded_flux_even_har_array, folded_flux_even_har_array_err,
                folded_flux_odd_har_array, folded_flux_odd_har_array_err], batch_data_values

# formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger()
# # while len(logger.handlers) > 0:
# #     logger.handlers.pop()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# dir = "/home/mdevora/scratch/ml/q1_q17/lcs"
# for filename in os.listdir(dir):
#     if 'input' in filename:
#         logging.info("Removing %s", filename)
#         os.remove(dir + '/' + filename)
#
# df = pd.read_csv("/home/mdevora/scratch/ml/q1_q17/injected_objects.csv", index_col=False)
# #df = shuffle(df)
# df.reset_index(inplace=True, drop=True)
# img = IatsonPlanetModelGenerator(df, "/home/mdevora/scratch/ml/q1_q17/lcs", 1, [11, 135, 2500, 500, 500, 500, 500, 500, 500],
#                                  {'fp': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0]}, from_arrays=False,
#                                  store_arrays=True)
# df_len = len(df)
# for i in range(0, df_len):
#     logging.info("Doing " + str(i) + " item of " + str(df_len))
#     img.__getitem__(i)
