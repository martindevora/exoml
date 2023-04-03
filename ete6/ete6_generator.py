import logging
from abc import abstractmethod

import matplotlib.pyplot as plt
import foldedleastsquares
import scipy
import timesynth
from lcbuilder.helper import LcbuilderHelper
from numpy.random import default_rng
from scipy import stats

from ml.encoding.time_position import value_encode_times
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd

from ml.functions.functions import sigmoid


class Ete6ModelGenerator(tf.keras.utils.Sequence):
    random_number_generator = default_rng()

    def __init__(self, zero_epsilon=1e-6):
        self.zero_epsilon = zero_epsilon

    def fold(self, time, period, epoch):
        return foldedleastsquares.fold(time, period, epoch) if len(time) > 0 else time

    def bin_interp(self, array, new_size):
        result = np.zeros((array.shape[1], new_size))
        for index, nested_array in enumerate(np.transpose(array)):
            try:
                nested_array = nested_array[nested_array > self.zero_epsilon]
                arr_interp = scipy.interpolate.interp1d(np.arange(nested_array.size), nested_array)
                result[index] = arr_interp(np.linspace(0, nested_array.size - 1, new_size))
            except Exception as e:
                a = 1
                # logging.exception("Error when interpolating, will create empty input array")
        return np.transpose(result)

    def bin_by_time(self, array, new_size, target):
        result = np.zeros((array.shape[1], new_size))
        result_err = np.zeros((array.shape[1], new_size))
        transposed_array = np.transpose(array)
        time_array = transposed_array[0]
        for index, nested_array in enumerate(transposed_array):
            if index > 0:
                mask = nested_array > self.zero_epsilon
                nested_array = nested_array[mask]
                masked_time_array = time_array[mask]
                try:
                    bin_means, bin_edges, binnumber = stats.binned_statistic(masked_time_array, nested_array,
                                                                             statistic='mean', bins=new_size)
                    bin_stds, _, _ = stats.binned_statistic(masked_time_array, nested_array, statistic='std',
                                                            bins=new_size)
                    if new_size > len(nested_array):
                        nans, x = self.nan_helper(bin_means)
                        bin_means[nans] = np.interp(x(nans), x(~nans), bin_means[~nans])
                        nans, x = self.nan_helper(bin_means)
                        bin_stds[nans] = np.interp(x(nans), x(~nans), bin_stds[~nans])
                    bin_width = (bin_edges[1] - bin_edges[0])
                    bin_centers = bin_edges[1:] - bin_width / 2
                    if np.max(result[0]) == 0:
                        result[0] = bin_centers
                        result_err[0] = np.full((new_size), 0.5)
                    # Store binning values for flux
                    result[index] = bin_means
                    result_err[index] = bin_stds if len(np.isnan(bin_stds)) == 0 else np.full((new_size), np.nanstd(bin_means))
                except Exception as e:
                    logging.exception("Error when interpolating for target " + str(target) + ", will create empty input array")
        return np.transpose(np.nan_to_num(result)), np.transpose(np.nan_to_num(result_err))

    @staticmethod
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    def _prepare_input_centroids(self, input_df):
        if len(input_df) > 0:
            _, ra_outliers_mask = LcbuilderHelper.clip_outliers(input_df['centroids_ra'].to_numpy(), 3)
            _, dec_outliers_mask = LcbuilderHelper.clip_outliers(input_df['centroids_dec'].to_numpy(), 3)
            outliers_mask = np.logical_or(ra_outliers_mask, dec_outliers_mask)
            input_df_clipped = input_df.loc[~outliers_mask]
            if len(input_df_clipped) > 0:
                min_value = np.nanmin(input_df['centroids_ra'])
                input_df['centroids_ra'] = input_df['centroids_ra'] - min_value
                input_df['centroids_ra'] = input_df['centroids_ra'] / np.nanmax(input_df['centroids_ra'])
                input_df['centroids_ra'] = input_df['centroids_ra'] - np.nanmedian(input_df['centroids_ra']) + 0.5
                min_value = np.nanmin(input_df['centroids_dec'])
                input_df['centroids_dec'] = input_df['centroids_dec'] - min_value
                input_df['centroids_dec'] = input_df['centroids_dec'] / np.nanmax(input_df['centroids_dec'])
                input_df['centroids_dec'] = input_df['centroids_dec'] - np.nanmedian(input_df['centroids_dec']) + 0.5
                input_df = input_df.fillna(self.zero_epsilon)
                input_df = input_df.replace(0.0, self.zero_epsilon)
                input_df = input_df.replace(0, self.zero_epsilon)
        return input_df

    def _prepare_input_og(self, input_df):
        if len(input_df) > 0:
            _, outliers_mask = LcbuilderHelper.clip_outliers(input_df['og_flux'].to_numpy(), 3)
            input_df_clipped = input_df.loc[~outliers_mask]
            if len(input_df_clipped) > 0:
                min_value = np.nanmin(input_df['og_flux'])
                input_df['og_flux'] = input_df['og_flux'] - min_value
                input_df['og_flux'] = input_df['og_flux'] / np.nanmax(input_df['og_flux'])
                input_df['og_flux'] = input_df['og_flux'] - np.nanmedian(input_df['og_flux']) + 0.5
            input_df = input_df.fillna(self.zero_epsilon)
            input_df = input_df.replace(0.0, self.zero_epsilon)
            input_df = input_df.replace(0, self.zero_epsilon)
        return input_df

    def _prepare_input_lc(self, input_df):
        time = input_df["#time"].to_numpy()
        # We don't want time encoding because time will be used for folding and then dropped
        #input_df["#time"] = value_encode_times(time - time[0])
        # red_noise_freq = self.random_number_generator.uniform(0.01, 0.1)
        # flux_std = np.std(input_df['flux'])
        # red_noise_sd = self.random_number_generator.uniform(0, flux_std * 3)
        # red_noise = timesynth.noise.RedNoise(std=red_noise_sd, tau=red_noise_freq)
        # timeseries_rn = np.zeros(len(time))
        # TODO red noise algorithm is too slow for long time series
        # timeseries_rn = []
        # for value in time:
        #     rn_value = red_noise.sample_next(value, None, None)
        #     rn_value = rn_value[0] if isinstance(rn_value, (list, np.ndarray)) else rn_value
        #     timeseries_rn = timeseries_rn + [rn_value]
        # timeseries_rn = np.array(timeseries_rn)
        # input_df['flux_0'] = input_df['flux_0'] + timeseries_rn
        input_df['flux_0'] = input_df['flux_0'] / 2
        input_df['flux_0'][input_df['flux_0'] > 1] = 1
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
        star_df = star_df.iloc[1:16, :].copy()
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

    @abstractmethod
    def class_weights(self):
        pass

# df = pd.read_csv("/mnt/DATA-2/ete6/injected_objects.csv", index_col=False)
# df = shuffle(df)
# df.reset_index(inplace=True, drop=True)
# img = IatsonModelGenerator(df, "/mnt/DATA-2/ete6/lcs", 20, [11, 2500, 500, 500, 500, 500, 500, 500])
# img.__getitem__(20)
