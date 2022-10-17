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
from ml.layers.transformer_classifier import ClassToken, AddPositionEmbs, TransformerClassifier
from ml.model.base_model import CategoricalPredictionSetStats
from ml.model.imbalanced_binary_model import ImbalancedBinaryModel


class IATSON_planet(ImbalancedBinaryModel):
    def __init__(self, class_ids, class_names, type_to_label, hyperparams, channels=12, name='IATSON_planet') -> None:
        super().__init__(name,
                         [11, 9 * 15, 2500, 500, 500, 500, 500, 500, 500],
                         class_ids,
                         type_to_label,
                         hyperparams)
        self.channels = channels

    def _get_flux_transformer_branch(self, name, transformer_blocks=6, transformer_heads=6):
        # TODO use binning error as channel
        flux_input = keras.Input(shape=(self.input_size[2], self.channels), name=name)
        flux_err_input = keras.Input(shape=(self.input_size[2], self.channels), name=name + "_err")
        flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_input)
        flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(flux_input_mod)
            flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(flux_err_input_mod)
        flux_input_final = keras.layers.concatenate([flux_input_mod, flux_err_input_mod], axis=2)
        transformer_input_size = 300
        transformer_output_size = 300
        kernel_size = 25
        classes = 16
        flux_branch = TransformerClassifier(transformer_input_size=transformer_input_size, patch_size=kernel_size,
                                            num_heads=transformer_heads, mlp_dim=transformer_output_size,
                                            dropout=self.hyperparams.dropout_rate,
                                            num_blocks=transformer_blocks, classes=classes)(flux_input_final)
        return flux_input, flux_err_input, flux_branch

    def _get_flux_conv_branch(self, name):
        # (time, flux, [detrended_flux1 ... detrended_flux5], flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        # TODO use binning error as channel
        leaky_relu_alpha = 0.01
        flux_input = keras.Input(shape=(self.input_size[2], self.channels), name=name)
        flux_err_input = keras.Input(shape=(self.input_size[2], self.channels), name=name + "_err")
        flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_input)
        flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std, name="gn_flux")(flux_input_mod)
            flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std, name="gn_flux_err")(flux_err_input_mod)
        flux_input_final = keras.layers.concatenate([flux_input_mod, flux_err_input_mod], axis=2)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_input_final)
        flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=128, kernel_size=9, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=10, strides=6)(flux_branch)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=4, strides=3)(flux_branch)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(flux_branch)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=2, strides=2)(flux_branch)
        return flux_input, flux_err_input, flux_branch

    def _get_flux_conv_model_branch(self):
        leaky_relu_alpha = 0.01
        flux_input, flux_err_input, flux_branch = self._get_flux_conv_branch("global_flux_branch")
        flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Dense(16, activation='linear')(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        flux_branch = keras.Model(inputs=[flux_input, flux_err_input], outputs=flux_branch)
        return flux_branch

    def _get_focus_flux_conv_branch(self, name):
        leaky_relu_alpha = 0.01
        # TODO use binning error as channel
        focus_flux_input = keras.Input(shape=(self.input_size[3], self.channels), name=name)
        focus_flux_err_input = keras.Input(shape=(self.input_size[3], self.channels), name=name + "_err")
        focus_flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_input)
        focus_flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            focus_flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_input_mod)
            focus_flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_err_input_mod)
        focus_flux_input_final = keras.layers.concatenate([focus_flux_input_mod, focus_flux_err_input_mod], axis=2)
        focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_input_final)
        # (time, flux, detrended_flux1... detrended_flux5, flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        focus_flux_branch = keras.layers.SpatialDropout1D(rate=self.hyperparams.dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=5, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.LayerNormalization()(focus_flux_branch)
        return focus_flux_input, focus_flux_err_input, focus_flux_branch

    def _get_focus_flux_transformer_branch(self, name, transformer_blocks=6, transformer_heads=6):
        # TODO use binning error as channel
        focus_flux_input = keras.Input(shape=(self.input_size[3], self.channels), name=name)
        focus_flux_err_input = keras.Input(shape=(self.input_size[3], self.channels), name=name + "_err")
        focus_flux_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_input)
        focus_flux_err_input_mod = keras.layers.Dropout(self.hyperparams.dropout_rate)(focus_flux_err_input)
        if self.hyperparams.white_noise_std is not None:
            focus_flux_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_input_mod)
            focus_flux_err_input_mod = keras.layers.GaussianNoise(stddev=self.hyperparams.white_noise_std)(focus_flux_err_input_mod)
        focus_flux_input_final = keras.layers.concatenate([focus_flux_input_mod, focus_flux_err_input_mod], axis=2)
        transformer_input_size = 100
        transformer_output_size = 100
        kernel_size = 5
        classes = 16
        flux_branch = TransformerClassifier(transformer_input_size=transformer_input_size, patch_size=kernel_size,
                              num_heads=transformer_heads, mlp_dim=transformer_output_size, dropout=self.hyperparams.dropout_rate,
                              num_blocks=transformer_blocks, classes=classes)(focus_flux_input_final)
        return focus_flux_input, focus_flux_err_input, flux_branch

    def _get_focus_flux_conv_model_branch(self):
        leaky_relu_alpha = 0.01
        odd_flux_input, odd_flux_err_input, odd_flux_branch = self._get_focus_flux_conv_branch("focus_odd_flux_branch")
        even_flux_input, even_flux_err_input, even_flux_branch = self._get_focus_flux_conv_branch("focus_even_flux_branch")
        harmonic_odd_flux_input, harmonic_odd_flux_err_input, harmonic_odd_flux_branch = \
            self._get_focus_flux_conv_branch("focus_harmonic_odd_flux_branch")
        harmonic_even_flux_input, harmonic_even_flux_err_input, harmonic_even_flux_branch = \
            self._get_focus_flux_conv_branch("focus_harmonic_even_flux_branch")
        subharmonic_odd_flux_input, subharmonic_odd_flux_err_input, subharmonic_odd_flux_branch = \
            self._get_focus_flux_conv_branch("focus_subharmonic_odd_flux_branch")
        subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_even_flux_branch = \
            self._get_focus_flux_conv_branch("focus_subharmonic_even_flux_branch")
        odd_flux_branch = keras.layers.Add()(
            [odd_flux_branch, harmonic_odd_flux_branch, subharmonic_odd_flux_branch])
        even_flux_branch = keras.layers.Add()(
            [even_flux_branch, harmonic_even_flux_branch, subharmonic_even_flux_branch])
        odd_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(odd_flux_branch)
        odd_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(even_flux_branch)
        even_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(even_flux_branch)
        odd_flux_branch = keras.layers.Dense(32, activation='linear')(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(32, activation='linear')(even_flux_branch)
        odd_flux_branch = keras.layers.LayerNormalization()(odd_flux_branch)
        even_flux_branch = keras.layers.LayerNormalization()(even_flux_branch)
        focus_flux_branch = keras.layers.Add()([odd_flux_branch, even_flux_branch])
        focus_flux_branch = keras.layers.Dense(24, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation='linear')(focus_flux_branch)
        input = [even_flux_input, even_flux_err_input, odd_flux_input, odd_flux_err_input,
                 subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_odd_flux_input,
                 subharmonic_odd_flux_err_input, harmonic_even_flux_input, harmonic_even_flux_err_input,
                 harmonic_odd_flux_input, harmonic_odd_flux_err_input]
        focus_flux_branch = keras.Model(inputs=input, outputs=focus_flux_branch)
        return focus_flux_branch

    def _get_focus_flux_transformer_model_branch(self, transformer_blocks=6, transformer_heads=6):
        leaky_relu_alpha = 0.01
        odd_flux_input, odd_flux_err_input, odd_flux_branch = self._get_focus_flux_transformer_branch("focus_odd_flux_branch")
        even_flux_input, even_flux_err_input, even_flux_branch = self._get_focus_flux_transformer_branch("focus_even_flux_branch")
        harmonic_odd_flux_input, harmonic_odd_flux_err_input, harmonic_odd_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_harmonic_odd_flux_branch", transformer_blocks,
                                                    transformer_heads)
        harmonic_even_flux_input, harmonic_even_flux_err_input, harmonic_even_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_harmonic_even_flux_branch", transformer_blocks,
                                                    transformer_heads)
        subharmonic_odd_flux_input, subharmonic_odd_flux_err_input, subharmonic_odd_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_subharmonic_odd_flux_branch", transformer_blocks,
                                                    transformer_heads)
        subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_even_flux_branch = \
            self._get_focus_flux_transformer_branch("focus_subharmonic_even_flux_branch", transformer_blocks,
                                                    transformer_heads)
        odd_flux_branch = keras.layers.Add()(
            [odd_flux_branch, harmonic_odd_flux_branch, subharmonic_odd_flux_branch])
        even_flux_branch = keras.layers.Add()(
            [even_flux_branch, harmonic_even_flux_branch, subharmonic_even_flux_branch])
        focus_flux_branch = keras.layers.Add()([odd_flux_branch, even_flux_branch])
        input = [even_flux_input, even_flux_err_input, odd_flux_input, odd_flux_err_input,
                 subharmonic_even_flux_input, subharmonic_even_flux_err_input, subharmonic_odd_flux_input,
                 subharmonic_odd_flux_err_input, harmonic_even_flux_input, harmonic_even_flux_err_input,
                 harmonic_odd_flux_input, harmonic_odd_flux_err_input]
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
        stellar_model_input = keras.Input(shape=(11, 1), name="stellar_model")
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="stellar-first")(
            stellar_model_input)
        stellar_model_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate, name="stellar-first-dropout-0.1")(
            stellar_model_branch)
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha),
                                                  name="stellar-refinement")(stellar_model_branch)
        # (TEFF, lum, 4 magnitudes, distance, radius, mass) * 15 stars
        stellar_neighbours_input = keras.Input(shape=(9 * 15, 1), name="stellar_neighbours_model")
        stellar_neighbours_branch = keras.layers.Dense(100, activation=LeakyReLU(leaky_relu_alpha),
                                                       name="stellar-neighbours-1")(
            stellar_neighbours_input)
        stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
                                                         name="stellar-neighbours-1-dropout")(
            stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dense(50, activation=LeakyReLU(leaky_relu_alpha),
                                                       name="stellar-neighbours-2")(
            stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
                                                         name="stellar-neighbours-2-dropout")(
            stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha),
                                                       name="stellar-neighbours_3")(
            stellar_neighbours_branch)
        flux_model_branch = self._get_flux_conv_model_branch()
        focus_flux_model_branch = self._get_focus_flux_conv_model_branch()
        final_branch = keras.layers.Concatenate(axis=1)(
            [stellar_model_branch, stellar_neighbours_branch, flux_model_branch.output, focus_flux_model_branch.output])
        final_branch = keras.layers.GlobalAveragePooling1D()(final_branch)
        final_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), name="final-dense1")(final_branch)
        final_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(final_branch)
        final_branch = keras.layers.Dense(32, activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(final_branch)
        final_branch = keras.layers.Dense(1, activation="sigmoid", name="logits")(final_branch)
        inputs = [stellar_model_input,
                  stellar_neighbours_input] + flux_model_branch.inputs + focus_flux_model_branch.inputs
        return inputs, final_branch

    def build_transformer(self, transformer_blocks=6, transformer_heads=6):
        leaky_relu_alpha = 0.01
        stellar_model_input = keras.Input(shape=(11, 1), name="stellar_model")
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="stellar-first")(
            stellar_model_input)
        stellar_model_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
                                                    name="stellar-first-dropout-0.1")(stellar_model_branch)
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="stellar-refinement")(
            stellar_model_branch)
        #(TEFF, lum, 4 magnitudes, distance, radius, mass) * 15 stars
        stellar_neighbours_input = keras.Input(shape=(9 * 15, 1), name="stellar_neighbours_model")
        stellar_neighbours_branch = keras.layers.Dense(100, activation=LeakyReLU(leaky_relu_alpha), name="stellar-neighbours-1")(
            stellar_neighbours_input)
        stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
                                                         name="stellar-neighbours-1-dropout-0.1")\
            (stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dense(50, activation=LeakyReLU(leaky_relu_alpha), name="stellar-neighbours-2")(
            stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate,
                                                         name="stellar-neighbours-2-dropout-0.1")\
            (stellar_neighbours_branch)
        stellar_neighbours_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha),
                                                       name="stellar-neighbours_3")(stellar_neighbours_branch)
        flux_model_input, flux_model_err_input, flux_model_branch = \
            self._get_flux_transformer_branch("global_flux_branch", transformer_blocks, transformer_heads)
        focus_flux_model_input, focus_flux_model_branch = \
            self._get_focus_flux_transformer_model_branch(transformer_blocks, transformer_heads)
        final_branch = keras.layers.Concatenate(axis=1)(
            [stellar_model_branch, stellar_neighbours_branch, flux_model_branch, focus_flux_model_branch])
        final_branch = keras.layers.GlobalAveragePooling1D()(final_branch)
        final_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), name="final-dense1")(final_branch)
        final_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(final_branch)
        final_branch = keras.layers.Dense(32, activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(final_branch)
        final_branch = keras.layers.Dense(1, activation="sigmoid", name="logits")(final_branch)
        inputs = [stellar_model_input, stellar_neighbours_input, flux_model_input, flux_model_err_input] + \
                 focus_flux_model_input
        return inputs, final_branch

    def load_training_set(self, **kwargs):
        training_dir = kwargs.get('training_dir')
        injected_objects_df = pd.read_csv(training_dir + "/injected_objects.csv", index_col=None)
        injected_objects_df = shuffle(injected_objects_df)
        injected_objects_df.reset_index(inplace=True, drop=True)
        return injected_objects_df

    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon):
        return IatsonPlanetModelGenerator(dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon)

    def predict(self, input):
        prediction = self.model(input, training=False)
        return self.predict_threshold([prediction])

    def predict_batch(self, inputs, expected_outputs=None, batch_size=20, cores=os.cpu_count() - 1):
        predictions = self.model.predict(inputs, use_multiprocessing=True, batch_size=batch_size, workers=cores,
                                         training=False)
        max_prediction_indexes, max_prediction_values = self.predict_threshold(predictions)
        if expected_outputs is not None:
            planet_stats = self.test_metrics(expected_outputs, max_prediction_indexes, max_prediction_values)
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

    def predict_test_set(self, training_dir, model_dir, batch_size=20, cores=os.cpu_count() - 1, test_set_limit=None,
                         zero_epsilon=1e-7):
        test_dataset = pd.read_csv(model_dir + '/test_dataset.csv')
        test_dataset = shuffle(test_dataset)
        if test_set_limit is not None:
            test_dataset = test_dataset.iloc[0:test_set_limit]
        lcs_dir = training_dir + '/lcs/'
        testing_batch_generator = IatsonPlanetModelGenerator(test_dataset, lcs_dir, batch_size, self.input_size,
                                                             zero_epsilon)
        expected_outputs = None
        for i in np.arange(0, len(test_dataset) // batch_size + 1, 1):
            input, expected_output = testing_batch_generator.__getitem__(i)
            expected_outputs = np.concatenate((expected_outputs, expected_output)) if expected_outputs is not None else expected_output
        max_prediction_indexes, max_prediction_values, prediction_stats = \
            self.predict_batch(testing_batch_generator, expected_outputs, batch_size, cores=cores)
        prediction_stats = prediction_stats[0]
        logging.info("Prediction stats for label %.0f: TP=%.0f, FP=%.0f, FN=%.0f, ACC=%.3f, PRE=%.3f, REC=%.3f", 0,
                     prediction_stats.tp, prediction_stats.fp, prediction_stats.fn, prediction_stats.accuracy,
                     prediction_stats.precision, prediction_stats.recall)


    def test_metrics(self, expected_outputs, max_prediction_indexes, max_prediction_values):
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
        precision = label_tp / (label_tp + label_fp)
        recall = label_tp / (label_tp + label_fn)
        return CategoricalPredictionSetStats(label_tp, label_fp, label_fn, accuracy, precision, recall)
