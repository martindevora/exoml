import matplotlib.pyplot as plt
import foldedleastsquares
import scipy
from numpy.random import default_rng
from scipy import stats

from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd

from ml.functions.functions import sigmoid


class IatsonModelGenerator(tf.keras.utils.Sequence):
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

    def bin_interp(self, array, new_size):
        result = np.zeros((array.shape[1], new_size))
        for index, nested_array in enumerate(np.transpose(array)):
            try:
                arr_interp = scipy.interpolate.interp1d(np.arange(nested_array.size), nested_array)
                result[index] = arr_interp(np.linspace(0, nested_array.size - 1, new_size))
            except Exception as e:
                a = 1
                # logging.exception("Error when interpolating, will create empty input array")
        return np.transpose(result)

    def bin_by_time(self, array, new_size):
        result = np.zeros((array.shape[1], new_size))
        transposed_array = np.transpose(array)
        for index, nested_array in enumerate(transposed_array):
            if index > 0:
                try:
                    bin_means, bin_edges, binnumber = stats.binned_statistic(transposed_array[0], nested_array,
                                                                             statistic='mean', bins=new_size)
                    bin_stds, _, _ = stats.binned_statistic(transposed_array[0], nested_array, statistic='std', bins=new_size)
                    bin_width = (bin_edges[1] - bin_edges[0])
                    bin_centers = bin_edges[1:] - bin_width / 2
                    if np.max(result[0]) == 0:
                        result[0] = bin_centers
                    result[index] = bin_means
                except Exception as e:
                    a = 1
                    # logging.exception("Error when interpolating, will create empty input array")
        return np.transpose(np.nan_to_num(result))

    def _prepare_input_lc(self, input_df):
        time = input_df["#time"].to_numpy()
        # We don't want time encoding because time will be used for folding and then dropped
        #input_df["#time"] = value_encode_times(time - time[0])
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
        star_df.drop('id', inplace=True, axis=1)
        star_df.drop('Unnamed: 0', inplace=True, axis=1)
        star_df.drop('ra', inplace=True, axis=1)
        star_df.drop('ra,', inplace=True, axis=1)
        star_df.drop('dec', inplace=True, axis=1)
        star_df.drop('dist_arcsec', inplace=True, axis=1)
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
        star_df['lum'] = sigmoid(star_df['lum'])
        star_df['logg'] = sigmoid(star_df['logg'])
        star_df['v'] = (30 - star_df['v']) / 40
        star_df['j'] = (30 - star_df['j']) / 40
        star_df['k'] = (30 - star_df['k']) / 40
        star_df['h'] = (30 - star_df['h']) / 40
        star_df = star_df.iloc[0]
        return star_df

    def _prepare_input_neighbour_stars(self, star_df):
        search_radius = 10
        pixel_size = 20.25
        max_distance = search_radius * pixel_size
        # Limiting the set to 15 stars, removing the first one, which is the target star
        star_df = star_df.iloc[1:16, :]
        star_df['Teff'] = star_df['Teff'].fillna(6000)
        star_df['radius'] = star_df['radius'].fillna(1)
        star_df['mass'] = star_df['mass'].fillna(1)
        star_df['lum'] = star_df['lum'].fillna(1)
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
        star_df['v'] = (30 - star_df['v']) / 40
        star_df['j'] = (30 - star_df['j']) / 40
        star_df['k'] = (30 - star_df['k']) / 40
        star_df['h'] = (30 - star_df['h']) / 40
        star_df['dist_arcsec'] = star_df['dist_arcsec'] / max_distance
        star_df['dist_arcsec'] = star_df['dist_arcsec'].fillna(max_distance)
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
        star_neighbours_array = np.empty((len(target_indexes), 9 * 15, 1))
        global_flux_array = np.empty((len(target_indexes), 2500, self.measurements_per_point))
        folded_flux_even_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        folded_flux_odd_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        folded_flux_even_subhar_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        folded_flux_odd_subhar_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        folded_flux_even_har_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        folded_flux_odd_har_array = np.empty((len(target_indexes), 500, self.measurements_per_point))
        batch_data_values = np.empty((len(target_indexes), len(self.TYPE_TO_LABEL)))
        i = 0
        for df_index, target_row in injected_objects_df.iterrows():
            target_id = int(target_row['TIC ID'])
            # target_id = 356110099
            # target_row = self.injected_objects_df[self.injected_objects_df['TIC ID'] == 356110099].iloc[0]
            leading_zeros_object_id = '{:09}'.format(target_id)
            lc_filename = str(leading_zeros_object_id) + '_lc.csv'
            star_filename = str(leading_zeros_object_id) + '_star.csv'
            star_neighbours_df = pd.read_csv(self.lcs_dir + '/' + star_filename, usecols=['Teff', 'lum', 'v', 'j',
                                                                           'k', 'h', 'radius', 'mass', 'dist_arcsec'],
                                             index_col=False)
            star_neighbours_df = self._prepare_input_neighbour_stars(star_neighbours_df)
            star_df = pd.read_csv(self.lcs_dir + '/' + star_filename, index_col=False)
            star_df = self._prepare_input_star(star_df)
            lc_df = pd.read_csv(self.lcs_dir + '/' + lc_filename, usecols=['#time', 'flux', 'flux_0', 'flux_1',
                                                                           'flux_2', 'flux_3', 'flux_4',
                                                                           'flux_err', 'centroid_x',
                                                                           'centroid_y', 'motion_x', 'motion_y',
                                                                           'bck_flux'], low_memory=True)
            lc_df = self._prepare_input_lc(lc_df)
            type = target_row['type']
            batch_data_values[i] = self.TYPE_TO_LABEL[type]
            not_null_times_args = np.argwhere(lc_df['#time'].to_numpy() > 0).flatten()
            lc_df = lc_df.iloc[not_null_times_args]
            period = target_row['period']
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
            lc_df_sorted_fold = lc_df_sorted_fold.sort_values(by=['#time'])
            global_flux_array[i] = self.bin_by_time(lc_df_sorted_fold.to_numpy(), self.input_sizes[1])
            # Focus flux even
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period, epoch)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_even_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[2])
            # Focus flux odd
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period, epoch + period / 2)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_odd_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[3])
            # if type != "none":
            #     self.__plot_df(lc_df_focus, type, "odd")
            #     self.__plot_input(folded_flux_odd_array[i], type, "odd")
            # Focus flux sub-harmonic even
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period / 2, epoch)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_even_subhar_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[4])
            # Focus flux sub-harmonic odd
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period / 2, epoch + period / 4)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_odd_subhar_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[5])
            # Focus flux harmonic even
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period * 2, epoch)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_even_har_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[6])
            # Focus flux harmonic odd
            lc_df_focus = lc_df.copy()
            lc_df_focus['#time'] = foldedleastsquares.fold(time, period * 2, epoch + period * 3 / 2)
            lc_df_focus = lc_df_focus.sort_values(by=['#time'])
            lc_df_focus = lc_df_focus[(lc_df_focus['#time'] > 0.5 - duration_to_period * 3) &
                                      (lc_df_focus['#time'] < 0.5 + duration_to_period * 3)]
            folded_flux_odd_har_array[i] = self.bin_by_time(lc_df_focus.to_numpy(), self.input_sizes[7])
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
            i = i + 1
        return [star_array, star_neighbours_array, global_flux_array, folded_flux_even_array, folded_flux_odd_array,
                folded_flux_even_subhar_array, folded_flux_odd_subhar_array, folded_flux_even_har_array,
                folded_flux_odd_har_array], batch_data_values

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
