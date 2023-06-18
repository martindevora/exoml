import logging
import os
from functools import reduce

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.activations import sigmoid
from keras.layers import LeakyReLU, Activation
from keras.utils import losses_utils
from keras.utils.generic_utils import to_list
from sklearn.utils import shuffle
from keras import backend as K

from iatson.iatson_planet_generator import IatsonPlanetModelGenerator
from ml.layers.dropout import AdaptiveStdDropout
from ml.layers.transformer_classifier import ClassToken, AddPositionEmbs, TransformerClassifier
from ml.metrics.auc import ThresholdAtPrecision
from ml.model.base_model import CategoricalPredictionSetStats
from ml.model.imbalanced_binary_model import ImbalancedBinaryModel


class IATSON_planet(ImbalancedBinaryModel):
    def __init__(self, class_ids, class_names, type_to_label, hyperparams, channels=2, name='IATSON_planet',
                 mode="all") -> None:
        super().__init__(name,
                         [11, 9 * 15, 300, 75, 75, 75, 75, 75, 75],
                         class_ids,
                         type_to_label,
                         hyperparams)
        self.channels = channels
        self.mode = mode

    def _get_flux_transformer_branch(self, name, transformer_blocks=2, transformer_heads=2):
        leaky_relu_alpha = 0.01
        # TODO use binning error as channel
        flux_input = keras.Input(shape=(self.input_size[2], self.channels), name=name)
        #flux_err_input = keras.Input(shape=(self.input_size[2], self.channels), name=name + "_err")
        #flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_input)
        flux_input_mod = flux_input
        #flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(flux_input_mod)
            #flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(flux_err_input_mod)
        #flux_input_final = keras.layers.concatenate([flux_input_mod, flux_err_input_mod], axis=2)
        flux_input_final = flux_input_mod
        transformer_input_size = self.input_size[2]
        transformer_output_size = self.input_size[2]
        kernel_size = 1
        classes = 128
        flux_branch = TransformerClassifier(transformer_input_size=transformer_input_size, patch_size=kernel_size,
                                            num_heads=transformer_heads, mlp_dim=transformer_output_size,
                                            hyperparams=self.hyperparams,
                                            num_blocks=transformer_blocks, classes=classes)(flux_input_final)
        flux_branch = keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                               activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        #return flux_input, flux_err_input, flux_branch
        return flux_input, flux_branch

    def _get_flux_conv_branch(self, name, channels=None):
        # (time, flux, [detrended_flux1 ... detrended_flux5], flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        # TODO use binning error as channel
        if not channels:
            channels = self.channels
        leaky_relu_alpha = 0.01
        flux_input = keras.Input(shape=(self.input_size[2], channels), name=name)
        flux_input_mod = flux_input
        #flux_err_input = keras.Input(shape=(self.input_size[2], channels), name=name + "_err")
        #flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_input)
        #flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std, name="gn_flux")(flux_input_mod)
            #flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std, name="gn_flux_err")(flux_err_input_mod)
        #flux_input_final = keras.layers.concatenate([flux_input_mod, flux_err_input_mod], axis=2)
        flux_input_final = flux_input_mod
        flux_branch = flux_input_final
        #flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=16, kernel_size=90, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=10, strides=1)(flux_branch)
        #flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=32, kernel_size=30, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=4, strides=1)(flux_branch)
        #flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=64, kernel_size=15, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=2)(flux_branch)
        #flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=2, strides=2)(flux_branch)
        return flux_input, flux_branch
        #return flux_input, flux_err_input, flux_branch

    def _get_flux_conv_model_branch(self):
        leaky_relu_alpha = 0.01
        flux_input, flux_branch = self._get_flux_conv_branch("global_flux_branch")
        flux_branch = keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                         activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        #flux_branch = keras.Model(inputs=[flux_input, flux_err_input], outputs=flux_branch)
        flux_branch = keras.Model(inputs=[flux_input], outputs=flux_branch)
        return flux_branch

    def _get_focus_flux_conv_branch(self, name, channels=None, additional_inputs=None):
        if not channels:
            channels = self.channels
        leaky_relu_alpha = 0.01
        # TODO use binning error as channel
        focus_flux_input = keras.Input(shape=(self.input_size[3], channels), name=name)
        focus_flux_input_mod = focus_flux_input
        #focus_flux_err_input = keras.Input(shape=(self.input_size[3], channels), name=name + "_err")
        #focus_flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_input)
        #focus_flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            focus_flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_input_mod)
            #focus_flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_err_input_mod)
        #focus_flux_input_final = keras.layers.concatenate([focus_flux_input_mod, focus_flux_err_input_mod], axis=2)
        focus_flux_input_final = focus_flux_input_mod
        focus_flux_branch = focus_flux_input_final
        #focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_input_final)
        # (time, flux, detrended_flux1... detrended_flux5, flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        #focus_flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=32, kernel_size=20, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=5, strides=1)(focus_flux_branch)
        #focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=64, kernel_size=10, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        #focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=1)(focus_flux_branch)
        focus_flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.spatial_dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        #focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=1)(focus_flux_branch)
        if additional_inputs is not None:
            focus_flux_branch = keras.layers.Flatten()(focus_flux_branch)
            focus_flux_branch = keras.layers.Concatenate(axis=1)([focus_flux_branch, keras.layers.Flatten()(additional_inputs)])
        focus_flux_branch = keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                               activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.LayerNormalization()(focus_flux_branch)
        return focus_flux_input, focus_flux_branch
        #return focus_flux_input, focus_flux_err_input, focus_flux_branch

    def _get_focus_flux_transformer_branch(self, name, transformer_blocks=6, transformer_heads=6, channels=None, additional_inputs=None):
        leaky_relu_alpha = 0.01
        # TODO use binning error as channel
        if channels is None:
            channels = self.channels
        focus_flux_input = keras.Input(shape=(self.input_size[3], channels), name=name)
        focus_flux_input_mod = focus_flux_input
        #focus_flux_err_input = keras.Input(shape=(self.input_size[3], self.channels), name=name + "_err")
        #focus_flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_input)
        #focus_flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            focus_flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_input_mod)
            #focus_flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_err_input_mod)
        focus_flux_input_final = focus_flux_input_mod
        #focus_flux_input_final = keras.layers.concatenate([focus_flux_input_mod, focus_flux_err_input_mod], axis=2)
        transformer_input_size = self.input_size[3]
        transformer_output_size = self.input_size[3]
        kernel_size = 1
        classes = 128
        flux_branch = TransformerClassifier(transformer_input_size=transformer_input_size, patch_size=kernel_size,
                              num_heads=transformer_heads, mlp_dim=transformer_output_size, hyperparams=self.hyperparams,
                              num_blocks=transformer_blocks, classes=classes)(focus_flux_input_final)
        if additional_inputs is not None:
            flux_branch = keras.layers.Flatten()(flux_branch)
            flux_branch = keras.layers.Concatenate(axis=1)([flux_branch, keras.layers.Flatten()(additional_inputs)])
        flux_branch = keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                               activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        return focus_flux_input, flux_branch
        #return focus_flux_input, focus_flux_err_input, flux_branch

    def _get_focus_flux_conv_model_branch(self, transit_params_input):
        leaky_relu_alpha = 0.01
        # odd_flux_input, odd_flux_err_input, odd_flux_branch = self._get_focus_flux_conv_branch("focus_odd_flux_branch")
        # even_flux_input, even_flux_err_input, even_flux_branch = self._get_focus_flux_conv_branch(
        #     "focus_even_flux_branch")
        # harmonic_odd_flux_input, harmonic_odd_flux_err_input, harmonic_odd_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_harmonic_odd_flux_branch")
        # harmonic_even_flux_input, harmonic_even_flux_err_input, harmonic_even_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_harmonic_even_flux_branch")
        # subharmonic_odd_flux_input, subharmonic_odd_flux_err_input, subharmonic_odd_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_subharmonic_odd_flux_branch")
        # subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_even_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_subharmonic_even_flux_branch")
        odd_flux_input, odd_flux_branch = self._get_focus_flux_conv_branch("focus_odd_flux_branch",
                                                                           additional_inputs=transit_params_input)
        even_flux_input, even_flux_branch = self._get_focus_flux_conv_branch("focus_even_flux_branch",
                                                                             additional_inputs=transit_params_input)
        harmonic_odd_flux_input, harmonic_odd_flux_branch = \
            self._get_focus_flux_conv_branch("focus_harmonic_odd_flux_branch")
        harmonic_even_flux_input, harmonic_even_flux_branch = \
            self._get_focus_flux_conv_branch("focus_harmonic_even_flux_branch")
        # subharmonic_odd_flux_input, subharmonic_odd_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_subharmonic_odd_flux_branch")
        # subharmonic_even_flux_input, subharmonic_even_flux_branch = \
        #     self._get_focus_flux_conv_branch("focus_subharmonic_even_flux_branch")
        flux_branch = keras.layers.Add()(
            [odd_flux_branch, even_flux_branch])#, subharmonic_odd_flux_branch])
        har_flux_branch = keras.layers.Add()(
            [harmonic_odd_flux_branch, harmonic_even_flux_branch])#, subharmonic_even_flux_branch])
        # odd_flux_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                      activation=LeakyReLU(leaky_relu_alpha))(odd_flux_branch)
        # odd_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(odd_flux_branch)
        # even_flux_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                       activation=LeakyReLU(leaky_relu_alpha))(even_flux_branch)
        # even_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(even_flux_branch)
        # odd_flux_branch = keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                      activation='linear')(odd_flux_branch)
        # even_flux_branch = keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                       activation='linear')(even_flux_branch)
        # odd_flux_branch = keras.layers.LayerNormalization()(odd_flux_branch)
        # even_flux_branch = keras.layers.LayerNormalization()(even_flux_branch)
        focus_flux_branch = keras.layers.Add()([flux_branch, har_flux_branch])
        # focus_flux_branch = keras.layers.Dense(24, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                        activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        # focus_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(focus_flux_branch)
        # focus_flux_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                        activation='linear')(focus_flux_branch)
        input = [even_flux_input, odd_flux_input, #subharmonic_even_flux_input, subharmonic_odd_flux_input,
                 harmonic_even_flux_input, harmonic_odd_flux_input]
        # input = [even_flux_input, even_flux_err_input, odd_flux_input, odd_flux_err_input,
        #          subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_odd_flux_input,
        #          subharmonic_odd_flux_err_input, harmonic_even_flux_input, harmonic_even_flux_err_input,
        #          harmonic_odd_flux_input, harmonic_odd_flux_err_input]
        return input, focus_flux_branch

    def _get_focus_flux_transformer_model_branch(self, transformer_blocks=6, transformer_heads=6, transit_params_input=None):
        leaky_relu_alpha = 0.01
        # odd_flux_input, odd_flux_err_input, odd_flux_branch = self._get_focus_flux_transformer_branch("focus_odd_flux_branch")
        # even_flux_input, even_flux_err_input, even_flux_branch = self._get_focus_flux_transformer_branch("focus_even_flux_branch")
        # harmonic_odd_flux_input, harmonic_odd_flux_err_input, harmonic_odd_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_harmonic_odd_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        # harmonic_even_flux_input, harmonic_even_flux_err_input, harmonic_even_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_harmonic_even_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        # subharmonic_odd_flux_input, subharmonic_odd_flux_err_input, subharmonic_odd_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_subharmonic_odd_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        # subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_even_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_subharmonic_even_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        odd_flux_input, odd_flux_branch = self._get_focus_flux_transformer_branch(
            "focus_odd_flux_branch", transformer_blocks, transformer_heads, additional_inputs=transit_params_input)
        even_flux_input, even_flux_branch = self._get_focus_flux_transformer_branch(
            "focus_even_flux_branch", transformer_blocks, transformer_heads, additional_inputs=transit_params_input)
        harmonic_odd_flux_input, harmonic_odd_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_harmonic_odd_flux_branch", transformer_blocks,
                                                    transformer_heads)
        harmonic_even_flux_input, harmonic_even_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_harmonic_even_flux_branch", transformer_blocks,
                                                    transformer_heads)
        # subharmonic_odd_flux_input, subharmonic_odd_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_subharmonic_odd_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        # subharmonic_even_flux_input, subharmonic_even_flux_branch = \
        #     self._get_focus_flux_transformer_branch("focus_subharmonic_even_flux_branch", transformer_blocks,
        #                                             transformer_heads)
        odd_flux_branch = keras.layers.Add()(
            [odd_flux_branch, harmonic_odd_flux_branch])#, subharmonic_odd_flux_branch])
        even_flux_branch = keras.layers.Add()(
            [even_flux_branch, harmonic_even_flux_branch])#, subharmonic_even_flux_branch])
        focus_flux_branch = keras.layers.Add()([odd_flux_branch, even_flux_branch])
        # input = [even_flux_input, even_flux_err_input, odd_flux_input, odd_flux_err_input,
        #          subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_odd_flux_input,
        #          subharmonic_odd_flux_err_input, harmonic_even_flux_input, harmonic_even_flux_err_input,
        #          harmonic_odd_flux_input, harmonic_odd_flux_err_input]
        input = [even_flux_input, odd_flux_input, #subharmonic_even_flux_input, subharmonic_odd_flux_input,
                 harmonic_even_flux_input, harmonic_odd_flux_input]
        return input, focus_flux_branch

    def _get_singletransit_tpf_model(self):
        video_image_width = 13
        video_image_height = 13
        video_image_channels = 1
        sequences_per_video = 100
        tpf_model_input = keras.Input(
            shape=(video_image_height, video_image_width, sequences_per_video, video_image_channels),
            name="tpf_input")
        tpf_model = keras.layers.SpatialDropout3D(rate=0.3)(tpf_model_input)
        tpf_model = keras.layers.Conv3D(filters=100, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.2)(tpf_model)
        tpf_model = keras.layers.Conv3D(filters=200, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.1)(tpf_model)
        tpf_model = keras.layers.MaxPooling3D(pool_size=(5, 5, 10), strides=(3, 3, 6), padding='same')(tpf_model)
        tpf_model = keras.layers.Dense(200, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(100, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(20, activation="relu")(tpf_model)
        tpf_model = keras.layers.Flatten()(tpf_model)
        return keras.Model(inputs=tpf_model_input, outputs=tpf_model)

    def _get_singletransit_motion_centroids_model(self):
        mc_input = keras.Input(
            shape=(100, 4),
            name="motion_centroids_input")
        mc_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                       padding='same')(mc_input)
        mc_model = keras.layers.SpatialDropout1D(rate=0.3)(mc_model)
        mc_model = keras.layers.Conv1D(filters=100, kernel_size=5, strides=5, activation="relu", use_bias=True,
                                       padding='same')(mc_model)
        mc_model = keras.layers.SpatialDropout1D(rate=0.2)(mc_model)
        mc_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(mc_model)
        mc_model = keras.layers.Dense(50, activation="relu")(mc_model)
        mc_model = keras.layers.Dense(20, activation="relu")(mc_model)
        mc_model = keras.layers.Flatten()(mc_model)
        return keras.Model(inputs=mc_input, outputs=mc_model)

    def _get_singletransit_bckflux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="bck_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def _get_singletransit_flux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="flux_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def _get_single_transit_model(self):
        tpf_branch = self._get_singletransit_tpf_model()
        mc_branch = self._get_singletransit_motion_centroids_model()
        bck_branch = self._get_singletransit_bckflux_model()
        flux_branch = self._get_singletransit_flux_model()
        final_branch = keras.layers.concatenate(
            [tpf_branch.output, mc_branch.output, bck_branch.output, flux_branch.output], axis=1)
        final_branch = keras.layers.Dense(64, activation="relu", name="final-dense1")(final_branch)
        final_branch = keras.layers.Dense(32, activation="relu", name="final-dense2")(final_branch)
        final_branch = keras.layers.Dense(1, activation="sigmoid", name="final-dense-sigmoid")(final_branch)
        inputs = tpf_branch.inputs + mc_branch.inputs + bck_branch.inputs + flux_branch.inputs
        model = keras.Model(inputs=inputs, outputs=final_branch, name="mnist_model")
        keras.utils.vis_utils.plot_model(model, "IATSON_model.png", show_shapes=True)

    def build(self, use_transformers=False, transformer_blocks=6, transformer_heads=6):
        inputs, final_branch = self.build_transformer(transformer_blocks, transformer_heads) \
            if use_transformers else self.build_convolutional()
        self.set_model(keras.Model(inputs=inputs, outputs=final_branch, name=self.name))
        return self

    def build_convolutional(self):
        leaky_relu_alpha = 0.01
        transit_params_input = keras.Input(shape=(6, 1), name="transit_params")
        offset_params_input = keras.Input(shape=(2, 1), name="offset_params")
        stellar_model_input = keras.Input(shape=(3, 1), name="stellar_model")
        stellar_model_branch = stellar_model_input
        # (TEFF, lum, 4 magnitudes, distance, radius, mass) * 15 stars
        # stellar_neighbours_input = keras.Input(shape=(9 * 15, 1), name="stellar_neighbours_model")
        # stellar_neighbours_branch = keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                activation=LeakyReLU(leaky_relu_alpha),
        #                                                name="stellar-neighbours-1")(
        #     stellar_neighbours_input)
        # stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
        #                                                  name="stellar-neighbours-1-dropout")(
        #     stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                activation=LeakyReLU(leaky_relu_alpha),
        #                                                name="stellar-neighbours-2")(
        #     stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
        #                                                  name="stellar-neighbours-2-dropout")(
        #     stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                activation=LeakyReLU(leaky_relu_alpha),
        #                                                name="stellar-neighbours_3")(
        #     stellar_neighbours_branch)
        flux_model_branch = self._get_flux_conv_model_branch()
        focus_flux_model_input, focus_flux_model_branch = self._get_focus_flux_conv_model_branch(transit_params_input)
        #centroids_input, centroids_err_input, centroids_branch = self._get_focus_flux_conv_branch("centroids_branch", 3)
        centroids_input, centroids_branch = self._get_focus_flux_conv_branch("centroids_branch", 3, offset_params_input)
        # centroids_inputs = [centroids_input, centroids_err_input]
        # og_input, og_err_input, og_branch = self._get_focus_flux_conv_branch("og_branch", 2)
        og_input, og_branch = self._get_focus_flux_conv_branch("og_branch", 2)
        # og_inputs = [og_input, og_err_input]
        # final_branch = keras.layers.Concatenate(axis=-1)(
        #     [stellar_model_branch,
        #      #stellar_neighbours_branch,
        #      flux_model_branch.output, focus_flux_model_branch,
        #      centroids_branch, og_branch])
        # final_branch = keras.layers.GlobalMaxPooling1D()(final_branch)
        final_branch = keras.layers.Concatenate(axis=-1)(
            [keras.layers.Flatten()(stellar_model_branch),
             # stellar_neighbours_branch,
             keras.layers.Flatten()(flux_model_branch.output),
             keras.layers.Flatten()(focus_flux_model_branch),
             keras.layers.Flatten()(centroids_branch),
             keras.layers.Flatten()(og_branch)])
        final_branch = keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate, name="final-dropout2")(final_branch)
        final_branch = keras.layers.Dense(250, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense3")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate, name="final-dropout3")(final_branch)
        final_branch = keras.layers.Dense(75, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense4")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate, name="final-dropout4")(final_branch)
        final_branch = keras.layers.Dense(35, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense5")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate, name="final-dropout5")(final_branch)
        final_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense6")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate, name="final-dropout6")(final_branch)
        final_branch = keras.layers.Dense(1, activation="sigmoid", name="logits")(final_branch)
        # inputs = [stellar_model_input, stellar_neighbours_input] + \
        #                  flux_model_branch.inputs + focus_flux_model_branch.inputs + \
        #                  centroids_inputs + og_inputs
        inputs = [transit_params_input, offset_params_input, stellar_model_input] + \
                 flux_model_branch.inputs + [focus_flux_model_input] + \
                 [centroids_input, og_input]
        return inputs, final_branch

    def build_transformer(self, transformer_blocks=6, transformer_heads=6):
        leaky_relu_alpha = 0.01
        transit_params_input = keras.Input(shape=(6, 1), name="transit_params")
        offset_params_input = keras.Input(shape=(2, 1), name="offset_params")
        stellar_model_input = keras.Input(shape=(3, 1), name="stellar_model")
        stellar_model_branch = stellar_model_input
        #(TEFF, lum, 4 magnitudes, distance, radius, mass) * 15 stars
        # stellar_neighbours_input = keras.Input(shape=(9 * 15, 1), name="stellar_neighbours_model")
        # stellar_neighbours_branch = keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                activation=LeakyReLU(leaky_relu_alpha), name="stellar-neighbours-1")(stellar_neighbours_input)
        # stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
        #                                                  name="stellar-neighbours-1-dropout-0.1")\
        #     (stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                activation=LeakyReLU(leaky_relu_alpha), name="stellar-neighbours-2")(
        #     stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
        #                                                  name="stellar-neighbours-2-dropout-0.1")\
        #     (stellar_neighbours_branch)
        # stellar_neighbours_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha),
        #                                                kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
        #                                                name="stellar-neighbours_3")(stellar_neighbours_branch)
        # flux_model_input, flux_model_err_input, flux_model_branch = \
        #     self._get_flux_transformer_branch("global_flux_branch", transformer_blocks, transformer_heads)
        flux_model_input, flux_model_branch = \
            self._get_flux_transformer_branch("global_flux_branch", transformer_blocks, transformer_heads)
        focus_flux_model_input, focus_flux_model_branch = \
            self._get_focus_flux_transformer_model_branch(transformer_blocks, transformer_heads, transit_params_input)
        #centroids_input, centroids_err_input, centroids_branch = self._get_focus_flux_conv_branch("centroids_branch", 3)
        centroids_input, centroids_branch = self._get_focus_flux_transformer_branch("centroids_branch",
                transformer_blocks, transformer_heads, additional_inputs=offset_params_input, channels=3)
        # og_input, og_err_input, og_branch = self._get_focus_flux_conv_branch("og_branch", 2)
        og_input, og_branch = self._get_focus_flux_transformer_branch("og_branch", transformer_blocks, transformer_heads, channels=2)
        final_branch = keras.layers.Concatenate(axis=-1, name="concat-final")(
            [keras.layers.Flatten()(stellar_model_branch),
             # stellar_neighbours_branch,
             keras.layers.Flatten()(flux_model_branch),
             keras.layers.Flatten()(focus_flux_model_branch),
             keras.layers.Flatten()(centroids_branch),
             keras.layers.Flatten()(og_branch)])
        final_branch = keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(final_branch)
        final_branch = keras.layers.Dense(250, kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense3")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(final_branch)
        final_branch = keras.layers.Dense(75, kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense4")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(final_branch)
        final_branch = keras.layers.Dense(35, kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense5")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(final_branch)
        final_branch = keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.L1L2(
            l1=self.hyperparams.l1_regularization, l2=self.hyperparams.l2_regularization),
                                          activation=LeakyReLU(leaky_relu_alpha), name="final-dense6")(final_branch)
        final_branch = AdaptiveStdDropout(rate=self.hyperparams.dropout_rate,
                                         max_rate=self.hyperparams.dropout_max_rate)(final_branch)
        final_branch = keras.layers.Dense(1, activation="sigmoid", name="logits")(final_branch)
        # inputs = [stellar_model_input, stellar_neighbours_input] + \
        #                  flux_model_branch.inputs + focus_flux_model_branch.inputs + \
        #                  centroids_inputs + og_inputs
        inputs = [transit_params_input, offset_params_input, stellar_model_input] + \
                 [flux_model_input, focus_flux_model_input] + \
                 [centroids_input, og_input]
        return inputs, final_branch

    def load_training_set(self, **kwargs):
        training_dir = kwargs.get('training_dir')
        injected_objects_df_q1_q17 = pd.read_csv(training_dir + "/q1_q17/injected_objects_tces.csv", index_col=None)
        injected_objects_df_ete6 = pd.read_csv(training_dir + "/ete6/injected_objects.csv", index_col=None)
        #injected_objects_df = injected_objects_df_q1_q17
        df = []
        if self.mode == "q1q17":
            df = [injected_objects_df_q1_q17]
        elif self.mode == 'ete6':
            df = [injected_objects_df_ete6]
        else:
            df = [injected_objects_df_ete6, injected_objects_df_q1_q17]
        injected_objects_df = pd.concat(df)
        injected_objects_df.reset_index(inplace=True, drop=True)
        return injected_objects_df

    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon, shuffle=True):
        return IatsonPlanetModelGenerator(dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon,
                                          shuffle_batch=shuffle)

    def predict(self, input):
        prediction = self.model(input, training=False)
        return self.predict_threshold([prediction])

    def predict_batch(self, inputs, expected_outputs=None, dataset=None, training_dir=None, plot_mismatches=False,
                      batch_size=20, cores=os.cpu_count() - 1, threshold=[0.5]):
        predictions = self.model.predict(inputs, use_multiprocessing=True, batch_size=batch_size, workers=cores)
        max_prediction_indexes, max_prediction_values = self.predict_threshold(predictions, threshold)
        if expected_outputs is not None:
            planet_stats = self.test_metrics(expected_outputs, max_prediction_indexes, max_prediction_values, predictions,
                                             dataset, training_dir=training_dir, plot_mismatches=plot_mismatches)
            return max_prediction_indexes, max_prediction_values, [planet_stats]
        else:
            return max_prediction_indexes, max_prediction_values

    def predict_threshold(self, predictions, threshold=[0.5]):
        max_prediction_indexes = np.array([np.argmax(prediction) for prediction in predictions])
        max_prediction_values = np.array([predictions[index][max_prediction_index]
                                 if predictions[index][max_prediction_index] > threshold[max_prediction_index]
                                 else -1
                                 for index, max_prediction_index in enumerate(max_prediction_indexes)])
        max_prediction_indexes[np.argwhere(max_prediction_values < 0).flatten()] = -1
        return max_prediction_indexes, max_prediction_values

    def predict_df(self, df, training_dir, batch_size=20, cores=os.cpu_count() - 1, zero_epsilon=1e-7,
                   thresholds=[0.5], plot_mismatches=False):
        expected_outputs = None
        testing_batch_generator = IatsonPlanetModelGenerator(df, training_dir, batch_size,
                                                             self.input_size,
                                                             self.type_to_label, zero_epsilon, from_arrays=True,
                                                             shuffle_batch=False)
        for i in np.arange(0, len(df) // batch_size + 1, 1):
            input, expected_output = testing_batch_generator.__getitem__(i)
            expected_outputs = np.concatenate((expected_outputs, expected_output)) if expected_outputs is not None else expected_output
        for threshold in thresholds:
            max_prediction_indexes, max_prediction_values, prediction_stats = \
                self.predict_batch(testing_batch_generator, expected_outputs, df, training_dir, plot_mismatches,
                                   batch_size, cores=cores, threshold=[threshold])
            prediction_stats = prediction_stats[0]
            logging.info("Prediction stats for label %.0f and threshold %s: TP=%.0f, FP=%.0f, FN=%.0f, ACC=%.3f, PRE=%.3f, REC=%.3f", 0,
                         threshold, prediction_stats.tp, prediction_stats.fp, prediction_stats.fn, prediction_stats.accuracy,
                         prediction_stats.precision, prediction_stats.recall)
            logging.info("Prediction stats @ k top predictions: \n%s", prediction_stats.k_df.to_string())
            mistmatches_df = prediction_stats.predictions_df[prediction_stats.predictions_df['expected_class'] !=
                                                             prediction_stats.predictions_df['predicted_class']]
            logging.info("Mismatches: \n%s", mistmatches_df.to_string())
            logging.info("Mismatches by types: \n%s", mistmatches_df
                         .groupby(by=['type', 'predicted_class'])['object_id'].count().to_string())

    def predict_test_set(self, training_dir, model_dir, batch_size=20, cores=os.cpu_count() - 1, test_set_limit=None,
                         zero_epsilon=1e-7, plot_mismatches=False, test_dataset_filename='test_dataset.csv',
                         thresholds=[0.5], logs=False):
        test_dataset = pd.read_csv(model_dir + '/' + test_dataset_filename)
        if test_set_limit is not None:
            test_dataset = test_dataset.iloc[0:test_set_limit]
        expected_outputs = None
        testing_batch_generator = IatsonPlanetModelGenerator(test_dataset, training_dir, batch_size,
                                                             self.input_size,
                                                             self.type_to_label, zero_epsilon, from_arrays=True,
                                                             shuffle_batch=False)
        for i in np.arange(0, len(test_dataset) // batch_size + 1, 1):
            input, expected_output = testing_batch_generator.__getitem__(i)
            expected_outputs = np.concatenate((expected_outputs, expected_output)) if expected_outputs is not None else expected_output
        threshold_to_results = {}
        for threshold in thresholds:
            max_prediction_indexes, max_prediction_values, prediction_stats = \
                self.predict_batch(testing_batch_generator, expected_outputs, test_dataset, training_dir, plot_mismatches,
                                   batch_size, cores=cores, threshold=[threshold])
            prediction_stats = prediction_stats[0]
            threshold_to_results[threshold] = prediction_stats
            if logs:
                logging.info("Prediction stats for label %.0f and threshold %s: TP=%.0f, FP=%.0f, FN=%.0f, ACC=%.3f, PRE=%.3f, REC=%.3f", 0,
                             threshold, prediction_stats.tp, prediction_stats.fp, prediction_stats.fn, prediction_stats.accuracy,
                             prediction_stats.precision, prediction_stats.recall)
                logging.info("Prediction stats @ k top predictions: \n%s", prediction_stats.k_df.to_string())
                mistmatches_df = prediction_stats.predictions_df[prediction_stats.predictions_df['expected_class'] !=
                                                                 prediction_stats.predictions_df['predicted_class']]
                logging.info("Mismatches: \n%s", mistmatches_df.to_string())
                logging.info("Mismatches by types: \n%s", mistmatches_df
                             .groupby(by=['type', 'predicted_class'])['object_id'].count().to_string())
        return threshold_to_results

    def predict_test_set_cv(self, training_dir, model_dir_prefix, batch_size=20, cores=os.cpu_count() - 1, test_set_limit=None,
                         zero_epsilon=1e-7, plot_mismatches=False, folds=10, thresholds=[0.5],
                            k_values=np.arange(100, 2200, 100)):
        fold_thresholds_to_results = {}
        prediction_dfs_thresholds = None
        predictions_df = None
        rp99s = np.zeros(folds)
        threshold_p99s = np.zeros(folds)
        for fold in np.arange(0, folds):
            model_dir = f"{model_dir_prefix}_{fold}/"
            self.load_model(model_dir)
            fold_thresholds_to_results[fold] = self.predict_test_set(training_dir, model_dir, batch_size, cores,
                                                    test_set_limit, zero_epsilon,
                                                    plot_mismatches, test_dataset_filename='validation_dataset.csv',
                                                    thresholds=thresholds, logs=False)
            predictions_df = fold_thresholds_to_results[fold][thresholds[0]].predictions_df if predictions_df is None \
                else pd.concat([predictions_df, fold_thresholds_to_results[fold][thresholds[0]].predictions_df])
        predictions_df_q1q17 = predictions_df[predictions_df['object_id'].str.contains("KIC")]
        predictions_df_ete6 = predictions_df[predictions_df['object_id'].str.contains("TIC")]
        predictions_df = predictions_df.sort_values(by=['prediction_value'], ascending=False)
        predictions_df_q1q17 = predictions_df_q1q17.sort_values(by=['prediction_value'], ascending=False)
        predictions_df_ete6 = predictions_df_ete6.sort_values(by=['prediction_value'], ascending=False)
        logging.info("----------------------------------COMPLETE SET METRICS ---------------------------------------")
        self.compute_cv_average_metrics(fold_thresholds_to_results, ['tp', 'fp', 'fn', 'precision', 'recall'], True)
        self.log_prediction_metrics(predictions_df, k_values, "K METRICS")
        self.compute_rp_thresholds(fold_thresholds_to_results, log=True)
        self.compute_rp_metrics(predictions_df, log=True)
        logging.info("----------------------------------Q1Q17 SET METRICS ---------------------------------------")
        self.log_prediction_metrics(predictions_df_q1q17, k_values, "K METRICS")
        self.compute_rp_metrics(predictions_df_q1q17, log=True)
        logging.info("----------------------------------Ete6 SET METRICS ---------------------------------------")
        self.log_prediction_metrics(predictions_df_ete6, k_values, "K METRICS")
        self.compute_rp_metrics(predictions_df_ete6, log=True)

    def log_prediction_metrics(self, df, k_range, message):
        logging.info(f"{message}")
        df = df.sort_values(by=['prediction_value'], ascending=False)
        for k in k_range:
            k_df = df.iloc[:k]
            k_tps = len(k_df.loc[(k_df['expected_class'] == k_df['predicted_class']) & (k_df['expected_class'] == 1)])
            k_fps = len(k_df.loc[(k_df['expected_class'] != k_df['predicted_class']) & (k_df['expected_class'] == 0)])
            k_precisions = k_tps / (k_tps + k_fps)
            logging.info("P@K%.0f = %.3f", k, k_precisions)
            logging.info("FPs@K%.0f = %.3f", k, k_fps)
        k = len(df[df['expected_class'] == 1])
        k_df = df.iloc[:k]
        k_tps = len(k_df.loc[(k_df['expected_class'] == k_df['predicted_class']) & (k_df['expected_class'] == 1)])
        k_fps = len(k_df.loc[(k_df['expected_class'] != k_df['predicted_class']) & (k_df['expected_class'] == 0)])
        k_precisions = k_tps / (k_tps + k_fps)
        logging.info("P@K%.0f = %.3f", k, k_precisions)
        logging.info("FPs@K%.0f = %.3f", k, k_fps)
        fns_df = df.loc[(df['expected_class'] != df['predicted_class']) & (df['expected_class'] == 1)]
        fns_df = fns_df.sort_values(by=['prediction_value'], ascending=True)
        logging.info("FNs DF: \n" + fns_df.to_string())
        fps_df = df.loc[(df['expected_class'] != df['predicted_class']) & (df['expected_class'] == 0)]
        fps_df = fps_df.sort_values(by=['prediction_value'], ascending=False)
        logging.info("FPs DF: \n" + fps_df.to_string())

    def compute_cv_average_metrics(self, fold_thresholds_to_results, metric_names, log=False):
        metric_fold_stats = {}
        metric_average_stats = {}
        for metric in metric_names:
            for fold in fold_thresholds_to_results:
                metric_val = getattr(fold_thresholds_to_results[fold][0.5], metric)
                if metric not in metric_fold_stats:
                    metric_fold_stats[metric] = {}
                metric_fold_stats[metric][fold] = metric_val
            metric_arr = [metric_val for metric_val in metric_fold_stats[metric].values()]
            metric_average_stats[metric] = (np.nanmean(metric_arr), np.nanstd(metric_arr))
        if log:
            logging.info(f"Average statistics:\n {metric_average_stats}")
        return metric_average_stats

    def compute_rp_thresholds(self, fold_thresholds_to_results, rp_value=0.99, thresholds=np.linspace(0, 1, 1000), log=False):
        selected_thresholds = []
        recalls = []
        for fold in fold_thresholds_to_results:
            threshold, recall = \
                self.compute_rp_metrics(fold_thresholds_to_results[fold][0.5].predictions_df, rp_value, thresholds, log)
            selected_thresholds = selected_thresholds + [threshold]
            recalls = recalls + [recall]
        logging.info(f"Average R@P{rp_value}: {np.nanmean(recalls)}")
        logging.info(f"R@P{rp_value} Thresholds: {selected_thresholds}")
        return selected_thresholds

    def compute_rp_metrics(self, df, rp_value=0.99, thresholds=np.linspace(0, 1, 1000), log=False):
        best_recall_threshold = 1
        best_recall = 0
        for threshold in thresholds:
            tp = len(df[(df['prediction_value'] >= threshold) & (df['expected_class'] == 1)])
            fp = len(df[(df['prediction_value'] >= threshold) & (df['expected_class'] == 0)])
            fn = len(df[(df['prediction_value'] < threshold) & (df['expected_class'] == 1)])
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            if precision >= 0.99 and recall > best_recall:
                best_recall_threshold = threshold
                best_recall = recall
        logging.info(f"Custom R@P{rp_value}: {best_recall}, {best_recall_threshold}")
        metric = tf.keras.metrics.RecallAtPrecision(0.99, num_thresholds=1000)
        metric.update_state(df['expected_class'].to_numpy(), df['prediction_value'].to_numpy())
        best_recall = metric.result().numpy()
        metric = ThresholdAtPrecision(0.99, num_thresholds=1000)
        metric.update_state(df['expected_class'].to_numpy(), df['prediction_value'].to_numpy())
        best_recall_threshold = metric.result().numpy()
        logging.info(f"Keras R@P{rp_value}: {best_recall}, {best_recall_threshold}")

        return best_recall_threshold, best_recall

    def test_metrics(self, expected_outputs, max_prediction_indexes, max_prediction_values, predictions,
                     dataset=None, training_dir=None, plot_mismatches=False):
        expected_outputs = expected_outputs.flatten()
        positive_preds = np.argwhere(max_prediction_indexes.flatten() != -1).flatten()
        negative_preds = np.argwhere(max_prediction_indexes.flatten() == -1).flatten()
        positive_outputs = np.argwhere(expected_outputs == 1).flatten()
        negative_outputs = np.argwhere(expected_outputs == 0).flatten()
        label_tp_indexes = np.intersect1d(positive_preds, positive_outputs)
        label_tn_indexes = np.intersect1d(negative_preds, negative_preds)
        label_fp_indexes = positive_preds[np.isin(positive_preds, negative_outputs)]
        label_fn_indexes = negative_preds[np.isin(negative_preds, positive_outputs)]
        label_tp = len(label_tp_indexes)
        label_tn = len(label_tn_indexes)
        label_fp = len(label_fp_indexes)
        label_fn = len(label_fn_indexes)
        accuracy = (label_tp + label_tn) / len(expected_outputs)
        precision = label_tp / (label_tp + label_fp + 1e-7)
        recall = label_tp / (label_tp + label_fn + 1e-7)
        max_predictions = np.array([np.max(prediction) for prediction in predictions])
        max_predictions_sort_args = np.flip(np.argsort(max_predictions))
        max_predictions_sort = max_predictions[max_predictions_sort_args]
        max_prediction_indexes_sort = max_prediction_indexes[max_predictions_sort_args]
        max_prediction_values_sort = max_prediction_values[max_predictions_sort_args]
        expected_outputs_sort = expected_outputs[max_predictions_sort_args]
        k_df = pd.DataFrame(columns=['k', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'tn', 'fn'])
        if 100 < len(expected_outputs):
            for k in np.arange(100, len(expected_outputs) if len(expected_outputs) < 2500 else 2500, 100):
                max_predictions_sort_k = max_predictions_sort[0:k]
                max_prediction_indexes_sort_k = max_prediction_indexes_sort[0:k]
                max_prediction_values_sort_k = max_prediction_values_sort[0:k]
                expected_outputs_sort_k = expected_outputs_sort[0:k]
                positive_preds = np.argwhere(max_prediction_indexes_sort_k.flatten() != -1).flatten()
                negative_preds = np.argwhere(max_prediction_indexes_sort_k.flatten() == -1).flatten()
                positive_outputs = np.argwhere(expected_outputs_sort_k == 1).flatten()
                negative_outputs = np.argwhere(expected_outputs_sort_k == 0).flatten()
                label_tp_indexes = np.intersect1d(positive_preds, positive_outputs)
                label_tn_indexes = np.intersect1d(negative_preds, negative_preds)
                label_fp_indexes = positive_preds[np.isin(positive_preds, negative_outputs)]
                label_fn_indexes = negative_preds[np.isin(negative_preds, positive_outputs)]
                label_tp = len(label_tp_indexes)
                label_tn = len(label_tn_indexes)
                label_fp = len(label_fp_indexes)
                label_fn = len(label_fn_indexes)
                accuracy = (label_tp + label_tn) / len(expected_outputs_sort_k)
                precision = label_tp / (label_tp + label_fp + 1e-7)
                recall = label_tp / (label_tp + label_fn + 1e-7)
                k_df = pd.concat([k_df, pd.DataFrame.from_dict(
                    {'k': [k], 'precision': [precision], 'recall': [recall], 'accuracy': [accuracy], 'tp': [label_tp],
                     'fp': [label_fp], 'tn': [label_tn], 'fn': [label_fn]}, orient='columns')], ignore_index=True)
        predictions_df = pd.DataFrame(columns=['object_id', 'type', 'expected_class', 'predicted_class',
                                              'prediction_value'])
        if dataset is not None:
            found_classes = [0 if prediction  == -1 else 1 for prediction in max_prediction_values]
            dataset = dataset.reset_index(drop=True)
            for i, row in dataset.iterrows():
                object_id = row['object_id']
                period = row['period']
                id = object_id + "_" + str(period)
                expected_tag = row['type']
                expected_output = expected_outputs[i]
                predicted_output = found_classes[i]
                predicted_value = predictions[i][max_prediction_indexes[i]]
                predictions_df = pd.concat([predictions_df, pd.DataFrame.from_dict(
                    {'object_id': [object_id + '_' + str(round(period, 2))], "type": [expected_tag],
                     "expected_class": [expected_output], "predicted_class": [predicted_output],
                     "prediction_value": [predicted_value]}, orient='columns')],
                                           ignore_index=True)
                if plot_mismatches:
                    logging.info("%s with label %s mismatched with value %s", id, expected_tag, predicted_value)
                    IatsonPlanetModelGenerator(dataset[(dataset['object_id'] == object_id) & (dataset['period'] == period)],
                                                                         training_dir, 1,
                                                                         self.input_size,
                                                                         self.type_to_label,
                                                                         from_arrays=True,
                                                                         shuffle_batch=False,
                                                                         plot_inputs=True).__getitem__(0)
        predictions_df = predictions_df.sort_values(by=['type', 'predicted_class'])
        return CategoricalPredictionSetStats(label_tp, label_fp, label_fn, accuracy, precision, recall, k_df,
                                             predictions_df)
# 2526 (6.824076075210719 %) items of class ['planet_transit']
# 2023-05-31 16:08:15 INFO     1857 (5.016749513723795 %) items of class ['planet']
# 2023-05-31 16:08:15 INFO     3559 (9.614761184352712 %) items of class ['fp']
# 2023-05-31 16:08:15 INFO     311 (0.8401772206613357 %) items of class ['fa']
# 2023-05-31 16:08:15 INFO     804 (2.172033715150205 %) items of class ['tce']
# 2023-05-31 16:08:15 INFO     2370 (6.402636697644263 %) items of class ['tce_secondary']
# 2023-05-31 16:08:15 INFO     1029 (2.7798789712556733 %) items of class ['tce_centroids_offset']
# 2023-05-31 16:08:15 INFO     3222 (8.7043440674303 %) items of class ['tce_source_offset']
# 2023-05-31 16:08:15 INFO     6980 (18.856710611627403 %) items of class ['tce_og']
# 2023-05-31 16:08:15 INFO     733 (1.9802247676680356 %) items of class ['tce_odd_even']
# 2023-05-31 16:08:15 INFO     12631 (34.12308191052518 %) items of class ['none']
# 2023-05-31 16:08:15 INFO     279 (0.7537281175707802 %) items of class ['EB']
# 2023-05-31 16:08:15 INFO     715 (1.9315971471795979 %) items of class ['bckEB']