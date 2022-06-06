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


class Ete6ModelGenerator(tf.keras.utils.Sequence):
    def __init__(self):
        pass

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
            input_df["centroid_x"][previous_jump_index:jumpIndex] = sigmoid(token - np.nanmedian(token))
            token = input_df["centroid_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_y"][previous_jump_index:jumpIndex] = sigmoid(token - np.nanmedian(token))
            token = input_df["motion_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_x"][previous_jump_index:jumpIndex] = sigmoid(token - np.nanmedian(token))
            token = input_df["motion_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_y"][previous_jump_index:jumpIndex] = sigmoid(token - np.nanmedian(token))
            token = input_df["bck_flux"][previous_jump_index:jumpIndex].to_numpy()
            input_df["bck_flux"][previous_jump_index:jumpIndex] = sigmoid(token - np.nanmedian(token))
            previous_jump_index = jumpIndex
        input_df = input_df.fillna(self.zero_epsilon)
        input_df = input_df.replace(0.0, self.zero_epsilon)
        input_df = input_df.replace(0, self.zero_epsilon)
        return input_df

    def _prepare_input_star(self, star_df):
        star_df.drop('id', inplace=True, axis=1)
        star_df.drop('Unnamed: 0', inplace=True, axis=1)
        star_df.drop('ra', inplace=True, axis=1)
        if pd.Series(['ra,']).isin(star_df.columns).any():
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
