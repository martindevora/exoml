import logging
import os.path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras import Model
from keras.layers import LeakyReLU
from keras.src.losses import binary_crossentropy, mean_squared_error

from exoml.haunter.haunter_generator import HaunterModelGenerator

from exoml.ml.model.base_model import BaseModel, HyperParams


G = 6.67430e-11  # Constante de gravitación universal en unidades SI (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Masa de la Tierra en kg
R_EARTH = 6.371e6   # Radio de la Tierra en metros
P_PASCAL = 1e5
RHO_CORE = 13000.0
RHO_MANTLE = 5000.0
RHO_WATER = 1000.0
RHO_GAS = 0.2
GAMMA_SCALING = 0.5  # Ajustable según calibración
PI = tf.constant(np.pi, dtype=tf.float32)


class Haunter(BaseModel):
    def __init__(self, hyperparameters: HyperParams, name='Haunter', input_size=6):
        super().__init__(name, input_size, None, None, hyperparameters)

    @staticmethod
    def load_data(dir):
        dataset = pd.read_csv(f"{dir}/models_for_ML_training_noduplicates.csv", sep=',')
        dataset = dataset.loc[(dataset['flag_physical'] == True) & (dataset['flag_convergence'] == 0)]
        dataset.dropna(inplace=True)
        dataset.loc[:, 'output_entropy_log10_1000bar_SI'] = np.log10(dataset.loc[:, 'output_entropy_1000bar_SI'])
        dataset.loc[:, 'output_log10_temperature_1000bar_K'] = np.log10(dataset.loc[:, 'output_temperature_1000bar_K'])
        return dataset

    def build(self):
        input = keras.Input(shape=(self.input_size,))
        nn = keras.layers.GaussianNoise(self.hyperparams.white_noise_std)(input)
        nn = keras.layers.Dense(1024, activation=LeakyReLU(0.01))(nn)
        nn = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Dense(1024, activation=LeakyReLU(0.01))(nn)
        nn = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Dense(4, activation="linear")(nn)
        #nn = keras.layers.Concatenate()([nn, input])
        self.set_model(Model(inputs=input, outputs=nn))
        return self

    def load_training_set(self, **kwargs):
        training_dir = kwargs.get('training_dir')
        dataset = Haunter.load_data(training_dir)
        self.input_means = tf.constant([
            dataset['input_mass_Mearth'].mean(),
            dataset['input_CMF'].mean(),
            dataset['input_Zenv'].mean(),
            dataset['input_Zwater_core'].mean(),
            dataset['input_Tsurf_K'].mean(),
            dataset['input_Psurf_bar'].mean()], dtype=tf.float32)
        self.input_stds = tf.constant([
            dataset['input_mass_Mearth'].std(),
            dataset['input_CMF'].std(),
            dataset['input_Zenv'].std(),
            dataset['input_Zwater_core'].std(),
            dataset['input_Tsurf_K'].std(),
            dataset['input_Psurf_bar'].std()], dtype=tf.float32)
        self.input_mins = tf.constant([
            dataset['input_mass_Mearth'].min(),
            dataset['input_CMF'].min(),
            dataset['input_Zenv'].min(),
            dataset['input_Zwater_core'].min(),
            dataset['input_Tsurf_K'].min(),
            dataset['input_Psurf_bar'].min()], dtype=tf.float32)
        self.input_maxs = tf.constant([
            dataset['input_mass_Mearth'].max(),
            dataset['input_CMF'].max(),
            dataset['input_Zenv'].max(),
            dataset['input_Zwater_core'].max(),
            dataset['input_Tsurf_K'].max(),
            dataset['input_Psurf_bar'].max()], dtype=tf.float32)
        self.output_means = tf.constant([
            dataset['output_radius_Rearth'].mean(),
            dataset['output_entropy_log10_1000bar_SI'].mean(),
            dataset['output_log10_f_s_SI'].mean(),
            dataset['output_log10_temperature_1000bar_K'].mean()], dtype=tf.float32)
        self.output_stds = tf.constant([
            dataset['output_radius_Rearth'].std(),
            dataset['output_entropy_log10_1000bar_SI'].std(),
            dataset['output_log10_f_s_SI'].std(),
            dataset['output_log10_temperature_1000bar_K'].std()], dtype=tf.float32)
        self.output_mins = tf.constant([
            dataset['output_radius_Rearth'].min(),
            dataset['output_entropy_log10_1000bar_SI'].min(),
            dataset['output_log10_f_s_SI'].min(),
            dataset['output_log10_temperature_1000bar_K'].min()], dtype=tf.float32)
        self.output_maxs = tf.constant([
            dataset['output_radius_Rearth'].max(),
            dataset['output_entropy_log10_1000bar_SI'].max(),
            dataset['output_log10_f_s_SI'].max(),
            dataset['output_log10_temperature_1000bar_K'].max()], dtype=tf.float32)
        dataset = self.add_standardized_columns(dataset, dataset)
        return dataset

    def instance_scaler(self):
        return StandardizeScaler()

    def add_standardized_columns(self, dataset, training_dataset):
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_mass_Mearth'])
        dataset.loc[:, 'input_mass_Mearth_norm'] = scaler.transform(dataset.loc[:, 'input_mass_Mearth'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_CMF'])
        dataset.loc[:, 'input_CMF_norm'] = scaler.transform(dataset.loc[:, 'input_CMF'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_Zenv'])
        dataset.loc[:, 'input_Zenv_norm'] = scaler.transform(dataset.loc[:, 'input_Zenv'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_Zwater_core'])
        dataset.loc[:, 'input_Zwater_core_norm'] = scaler.transform(dataset.loc[:, 'input_Zwater_core'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_Tsurf_K'])
        dataset.loc[:, 'input_Tsurf_K_norm'] = scaler.transform(dataset.loc[:, 'input_Tsurf_K'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'input_Psurf_bar'])
        dataset.loc[:, 'input_Psurf_bar_norm'] = scaler.transform(dataset.loc[:, 'input_Psurf_bar'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_radius_Rearth'])
        dataset.loc[:, 'output_radius_Rearth_norm'] = scaler.transform(
            dataset.loc[:, 'output_radius_Rearth'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'])
        dataset.loc[:, 'output_entropy_log10_1000bar_SI_norm'] = scaler.transform(dataset.loc[:, 'output_entropy_log10_1000bar_SI'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_log10_f_s_SI'])
        dataset.loc[:, 'output_log10_f_s_SI_norm'] = scaler.transform(dataset.loc[:, 'output_log10_f_s_SI'].to_numpy())
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'])
        dataset.loc[:, 'output_log10_temperature_1000bar_K_norm'] = scaler.transform(dataset.loc[:, 'output_log10_temperature_1000bar_K'].to_numpy())
        logging.info("Mass " + str(training_dataset.loc[:, 'input_mass_Mearth'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_mass_Mearth'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_mass_Mearth'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_mass_Mearth'].max()))
        logging.info("CMF " + str(training_dataset.loc[:, 'input_CMF'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_CMF'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_CMF'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_CMF'].max()))
        logging.info("Zenv " + str(training_dataset.loc[:, 'input_Zenv'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Zenv'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Zenv'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Zenv'].max()))
        logging.info("Zwater " + str(training_dataset.loc[:, 'input_Zwater_core'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Zwater_core'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Zwater_core'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Zwater_core'].max()))
        logging.info("Tsurf " + str(training_dataset.loc[:, 'input_Tsurf_K'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Tsurf_K'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Tsurf_K'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Tsurf_K'].max()))
        logging.info("Psurf " + str(training_dataset.loc[:, 'input_Psurf_bar'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Psurf_bar'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Psurf_bar'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Psurf_bar'].max()))
        logging.info("Radius " + str(training_dataset.loc[:, 'output_radius_Rearth'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_radius_Rearth'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_radius_Rearth'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_radius_Rearth'].max()))
        logging.info("Entropy " + str(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].max()))
        logging.info("F_S " + str(training_dataset.loc[:, 'output_log10_f_s_SI'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_log10_f_s_SI'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_log10_f_s_SI'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_log10_f_s_SI'].max()))
        logging.info("Log10 Temperature at 1000 bar " + str(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].max()))
        return dataset

    def add_destandardized_columns(self, dataset, training_dataset, predictions):
        logging.info("Mass " + str(training_dataset.loc[:, 'input_mass_Mearth'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_mass_Mearth'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_mass_Mearth'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_mass_Mearth'].max()))
        logging.info(
            "CMF " + str(training_dataset.loc[:, 'input_CMF'].mean()) + ' +- ' + str(
                training_dataset.loc[:, 'input_CMF'].std()) +
            ' ' + str(training_dataset.loc[:, 'input_CMF'].min()) + ' ' + str(
                training_dataset.loc[:, 'input_CMF'].max()))
        logging.info(
            "Zenv " + str(training_dataset.loc[:, 'input_Zenv'].mean()) + ' +- ' + str(
                training_dataset.loc[:, 'input_Zenv'].std()) +
            ' ' + str(training_dataset.loc[:, 'input_Zenv'].min()) + ' ' + str(
                training_dataset.loc[:, 'input_Zenv'].max()))
        logging.info("Zwater " + str(training_dataset.loc[:, 'input_Zwater_core'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Zwater_core'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Zwater_core'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Zwater_core'].max()))
        logging.info("Tsurf " + str(training_dataset.loc[:, 'input_Tsurf_K'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Tsurf_K'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Tsurf_K'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Tsurf_K'].max()))
        logging.info("Psurf " + str(training_dataset.loc[:, 'input_Psurf_bar'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'input_Psurf_bar'].std()) +
                     ' ' + str(training_dataset.loc[:, 'input_Psurf_bar'].min()) + ' ' + str(
            training_dataset.loc[:, 'input_Psurf_bar'].max()))
        logging.info("Radius " + str(training_dataset.loc[:, 'output_radius_Rearth'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_radius_Rearth'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_radius_Rearth'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_radius_Rearth'].max()))
        logging.info("Entropy " + str(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].mean()) + ' +- ' + str(
            training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].std()) +
                     ' ' + str(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].min()) + ' ' + str(
            training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'].max()))
        logging.info(
            "F_S " + str(training_dataset.loc[:, 'output_log10_f_s_SI'].mean()) + ' +- ' + str(
                training_dataset.loc[:, 'output_log10_f_s_SI'].std()) +
            ' ' + str(training_dataset.loc[:, 'output_log10_f_s_SI'].min()) + ' ' + str(
                training_dataset.loc[:, 'output_log10_f_s_SI'].max()))
        logging.info(
            "Log10 Temperature K at 1000 bar " + str(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].mean()) + ' +- ' + str(
                training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].std()) +
            ' ' + str(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].min()) + ' ' + str(
                training_dataset.loc[:, 'output_log10_temperature_1000bar_K'].max()))
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_radius_Rearth'])
        dataset.loc[:, 'output_radius_Rearth_pred'] = scaler.inverse_transform(predictions[:, 0])
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_entropy_log10_1000bar_SI'])
        dataset.loc[:, 'output_entropy_log10_1000bar_SI_pred'] = scaler.inverse_transform(predictions[:, 1])
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_log10_f_s_SI'])
        dataset.loc[:, 'output_log10_f_s_SI_pred'] = scaler.inverse_transform(predictions[:, 2])
        scaler = self.instance_scaler()
        scaler.fit(training_dataset.loc[:, 'output_log10_temperature_1000bar_K'])
        dataset.loc[:, 'output_log10_temperature_1000bar_K_pred'] = scaler.inverse_transform(predictions[:, 3])
        return dataset

    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon=1e-7, shuffle=True):
        return HaunterModelGenerator(dataset, batch_size, input_sizes, zero_epsilon, shuffle=shuffle)

    def instance_loss_accuracy(self):
        # loss = tf.keras.losses.MeanAbsolutePercentageError(reduction=losses_utils.ReductionV2.AUTO)
        loss = tf.keras.losses.MeanSquaredError()
        #loss = self.combined_loss
        return loss, None

    def instance_metrics(self):
        return [self.mae_radius, self.mae_entropy, self.mae_fs, self.mae_temperature]

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        return []

    def load_model(self, dir, custom_objects={}, compile=False):
        custom_objects['LeakyReLU'] = LeakyReLU
        custom_objects['mse_radius'] = self.mse_radius
        custom_objects['mse_entropy'] = self.mse_entropy
        custom_objects['mse_fs'] = self.mse_fs
        custom_objects['mae_radius'] = self.mae_radius
        custom_objects['mae_entropy'] = self.mae_entropy
        custom_objects['mae_fs'] = self.mae_fs
        custom_objects['mae_temperature'] = self.mae_temperature
        logging.info("Loading model from %s", dir)
        self.set_model(tf.keras.models.load_model(dir, compile=compile, custom_objects=custom_objects))
        return self

    def predict_validation(self, model_dir, dir, batch_size=128):
        dataset_training = pd.read_csv(f"{model_dir}/train_dataset.csv")
        dataset_validation = pd.read_csv(f"{model_dir}/validation_dataset.csv")
        dataset_standardization = pd.concat([dataset_training, dataset_validation])
        dataset_validation = self.add_standardized_columns(dataset_validation, dataset_standardization)
        predict_batch_generator = self.instance_generator(dataset_validation, None, batch_size, self.input_size, None,
                                                          shuffle=False)
        predictions = self.model.predict(predict_batch_generator, batch_size=batch_size)
        dataset_validation['output_radius_Rearth_norm_pred'] = predictions[:, 0]
        dataset_validation['output_entropy_log10_1000bar_SI_norm_pred'] = predictions[:, 1]
        dataset_validation['output_log10_f_s_SI_norm_pred'] = predictions[:, 2]
        dataset_validation['output_log10_temperature_1000bar_K_norm_pred'] = predictions[:, 3]
        dataset_validation = self.add_destandardized_columns(dataset_validation, dataset_standardization, predictions)
        dataset_validation['output_radius_Rearth_pred_diff'] = np.abs(
            dataset_validation['output_radius_Rearth_pred'] - dataset_validation['output_radius_Rearth'])
        dataset_validation['output_entropy_log10_1000bar_SI_pred_diff'] = np.abs(
            dataset_validation['output_entropy_log10_1000bar_SI_pred'] - dataset_validation['output_entropy_log10_1000bar_SI'])
        dataset_validation['output_log10_f_s_SI_pred_diff'] = np.abs(
            dataset_validation['output_log10_f_s_SI_pred'] - dataset_validation['output_log10_f_s_SI'])
        dataset_validation['output_log10_temperature_1000bar_K_pred_diff'] = np.abs(dataset_validation['output_log10_temperature_1000bar_K_pred'] -
                                                                               dataset_validation['output_log10_temperature_1000bar_K'])
        dataset_validation['output_radius_Rearth_pred_diff_ratio'] = np.abs(
            dataset_validation['output_radius_Rearth_pred'] - dataset_validation['output_radius_Rearth']) / \
                                                                     dataset_validation['output_radius_Rearth']
        dataset_validation['output_entropy_log10_1000bar_SI_pred_diff_ratio'] = np.abs(
            dataset_validation['output_entropy_log10_1000bar_SI_pred'] - dataset_validation['output_entropy_log10_1000bar_SI']) / \
                                                                  dataset_validation['output_entropy_log10_1000bar_SI']
        dataset_validation['output_log10_f_s_SI_pred_diff_ratio'] = np.abs(
            dataset_validation['output_log10_f_s_SI_pred'] - dataset_validation['output_log10_f_s_SI']) / np.abs(
            dataset_validation['output_log10_f_s_SI'])
        dataset_validation['output_log10_temperature_1000bar_K_pred_diff_ratio'] = np.abs(
            dataset_validation['output_log10_temperature_1000bar_K_pred'] - dataset_validation['output_log10_temperature_1000bar_K']) / np.abs(
            dataset_validation['output_log10_temperature_1000bar_K'])
        results_df = dataset_validation[['output_radius_Rearth', 'output_radius_Rearth_norm',
                                         'output_radius_Rearth_pred', 'output_radius_Rearth_norm_pred',
                                         'output_radius_Rearth_pred_diff', 'output_radius_Rearth_pred_diff_ratio',
                                         'output_entropy_log10_1000bar_SI', 'output_entropy_log10_1000bar_SI_norm',
                                         'output_entropy_log10_1000bar_SI_pred', 'output_entropy_log10_1000bar_SI_norm_pred',
                                         'output_entropy_log10_1000bar_SI_pred_diff', 'output_entropy_log10_1000bar_SI_pred_diff_ratio',
                                         'output_log10_f_s_SI', 'output_log10_f_s_SI_norm',
                                         'output_log10_f_s_SI_pred', 'output_log10_f_s_SI_norm_pred',
                                         'output_log10_f_s_SI_pred_diff', 'output_log10_f_s_SI_pred_diff_ratio',
                                         'output_log10_temperature_1000bar_K', 'output_log10_temperature_1000bar_K_norm',
                                         'output_log10_temperature_1000bar_K_pred', 'output_log10_temperature_1000bar_K_norm_pred',
                                         'output_log10_temperature_1000bar_K_pred_diff', 'output_log10_temperature_1000bar_K_pred_diff_ratio',]]
        logging.info("Results validation: \n%s", results_df.to_string())
        logging.info("Validation radius diff mean: " + str(
            dataset_validation.loc[:, 'output_radius_Rearth_pred_diff'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_radius_Rearth_pred_diff'].std()))
        logging.info("Validation output_entropy_log10_1000bar_SI diff mean: " + str(
            dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff'].std()))
        logging.info("Validation F_S diff mean: " + str(
            dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff'].std()))
        logging.info("Validation Log10 Temperature diff mean: " + str(
            dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff'].std()))
        logging.info("Validation radius diff ratio median: " + str(
            dataset_validation.loc[:, 'output_radius_Rearth_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_validation.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("Validation output_entropy_log10_1000bar_SI diff ratio median: " + str(
            dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("Validation output_log10_f_s_SI diff ratio median: " + str(
            dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].std()))
        logging.info("Validation Log10 Temperature diff ratio median: " + str(
            dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].std()))
        logging.info("Validation radius diff ratio mean: " + str(
            dataset_validation.loc[:, 'output_radius_Rearth_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("Validation output_entropy_log10_1000bar_SI diff ratio mean: " + str(
            dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("Validation output_log10_f_s_SI diff ratio mean: " + str(
           dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].mean()) + '+-' +
                    str(dataset_validation.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].std()))
        logging.info("Validation Log10 Temperature diff ratio mean: " + str(
            dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_validation.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].std()))
        return dataset_validation

    def predict_test(self, model_dir, test_csv, batch_size=128):
        dataset_training = pd.read_csv(f"{model_dir}/train_dataset.csv")
        dataset_validation = pd.read_csv(f"{model_dir}/validation_dataset.csv")
        dataset_standardization = pd.concat([dataset_training, dataset_validation])
        dataset_test = pd.read_csv(test_csv)
        dataset_test = dataset_test.loc[(dataset_test['flag_physical'] == True) & (dataset_test['flag_convergence'] == 0)]
        dataset_test.dropna(inplace=True)
        dataset_test.loc[:, 'output_entropy_log10_1000bar_SI'] = np.log10(dataset_test.loc[:, 'output_entropy_1000bar_SI'])
        dataset_test.loc[:, 'output_log10_temperature_1000bar_K'] = np.log10(dataset_test.loc[:, 'output_temperature_1000bar_K'])
        dataset_test = self.add_standardized_columns(dataset_test, dataset_standardization)
        predict_batch_generator = self.instance_generator(dataset_test, None, batch_size, self.input_size, None,
                                                          shuffle=False)
        predictions = self.model.predict(predict_batch_generator, batch_size=batch_size)
        dataset_test['output_radius_Rearth_norm_pred'] = predictions[:, 0]
        dataset_test['output_entropy_log10_1000bar_SI_norm_pred'] = predictions[:, 1]
        dataset_test['output_log10_f_s_SI_norm_pred'] = predictions[:, 2]
        dataset_test['output_log10_temperature_1000bar_K_norm_pred'] = predictions[:, 3]
        dataset_test = self.add_destandardized_columns(dataset_test, dataset_standardization, predictions)
        dataset_test['output_radius_Rearth_pred_diff'] = np.abs(
            dataset_test['output_radius_Rearth_pred'] - dataset_test['output_radius_Rearth'])
        dataset_test['output_entropy_log10_1000bar_SI_pred_diff'] = np.abs(
            dataset_test['output_entropy_log10_1000bar_SI_pred'] - dataset_test['output_entropy_log10_1000bar_SI'])
        dataset_test['output_log10_f_s_SI_pred_diff'] = np.abs(
            dataset_test['output_log10_f_s_SI_pred'] - dataset_test['output_log10_f_s_SI'])
        dataset_test['output_log10_temperature_1000bar_K_pred_diff'] = np.abs(
            dataset_test['output_log10_temperature_1000bar_K_pred'] - dataset_test['output_log10_temperature_1000bar_K'])
        dataset_test['output_radius_Rearth_pred_diff_ratio'] = np.abs(
            dataset_test['output_radius_Rearth_pred'] - dataset_test['output_radius_Rearth']) / \
            dataset_test['output_radius_Rearth']
        dataset_test['output_entropy_log10_1000bar_SI_pred_diff_ratio'] = np.abs(
            dataset_test['output_entropy_log10_1000bar_SI_pred'] - dataset_test['output_entropy_log10_1000bar_SI']) / \
            dataset_test['output_entropy_log10_1000bar_SI']
        dataset_test['output_log10_f_s_SI_pred_diff_ratio'] = np.abs(
            dataset_test['output_log10_f_s_SI_pred'] - dataset_test['output_log10_f_s_SI']) / np.abs(
            dataset_test['output_log10_f_s_SI'])
        dataset_test['output_log10_temperature_1000bar_K_pred_diff_ratio'] = np.abs(
            dataset_test['output_log10_temperature_1000bar_K_pred'] - dataset_test['output_log10_temperature_1000bar_K']) / np.abs(
            dataset_test['output_log10_temperature_1000bar_K'])
        logging.info("Test radius diff mean: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff'].std()))
        logging.info("Test entropy diff mean: " + str(
            dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff'].std()))
        logging.info("Test f_s diff mean: " + str(
            dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff'].std()))
        logging.info("Test temperature diff mean: " + str(
            dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff'].std()))
        logging.info("Test radius diff ratio median: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("Test entropy diff ratio median: " + str(
            dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("Test f_s diff ratio median: " + str(
            dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].std()))
        logging.info("Test temperature diff ratio median: " + str(
            dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].std()))
        logging.info("Test radius diff ratio mean: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("Test entropy diff ratio mean: " + str(
            dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_log10_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("Test f_s diff ratio mean: " + str(
            dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_f_s_SI_pred_diff_ratio'].std()))
        logging.info("Test temperature diff ratio mean: " + str(
            dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_log10_temperature_1000bar_K_pred_diff_ratio'].std()))
        return dataset_test

    def predict_test_cv(self, model_dir, test_csv, folds=5, batch_size=128):
        # Load and preprocess test set
        dataset_test = pd.read_csv(test_csv)
        dataset_test = dataset_test.loc[(dataset_test['flag_physical'] == True) & (dataset_test['flag_convergence'] == 0)]
        dataset_test.dropna(inplace=True)

        # Save ground truth before predict_cv replaces the DataFrame
        truth = dataset_test[['output_radius_Rearth', 'output_entropy_1000bar_SI',
                               'output_log10_f_s_SI', 'output_temperature_1000bar_K']].copy()

        # Delegate fold ensemble predictions to predict_cv
        input_cols = ['input_mass_Mearth', 'input_CMF', 'input_Zenv', 'input_Zwater_core',
                      'input_Tsurf_K', 'input_Psurf_bar']
        dataset_test = self.predict_cv(dataset_test[input_cols], model_dir, folds=folds, batch_size=batch_size)

        # Reattach ground truth
        dataset_test = pd.concat([dataset_test, truth], axis=1)

        # Compute diffs against ground truth
        dataset_test['output_radius_Rearth_pred_diff'] = np.abs(
            dataset_test['radius_Rearth_pred'] - dataset_test['output_radius_Rearth'])
        dataset_test['output_radius_Rearth_pred_diff_ratio'] = (
            dataset_test['output_radius_Rearth_pred_diff'] / dataset_test['output_radius_Rearth'])

        dataset_test['output_entropy_1000bar_SI_pred_diff'] = np.abs(
            dataset_test['entropy_1000bar_SI_pred'] - dataset_test['output_entropy_1000bar_SI'])
        dataset_test['output_entropy_1000bar_SI_pred_diff_ratio'] = (
            dataset_test['output_entropy_1000bar_SI_pred_diff'] / dataset_test['output_entropy_1000bar_SI'])

        dataset_test['output_f_s_SI_pred_diff'] = np.abs(
            dataset_test['f_s_SI_pred'] - 10 ** dataset_test['output_log10_f_s_SI'])
        dataset_test['output_f_s_SI_pred_diff_ratio'] = (
            dataset_test['output_f_s_SI_pred_diff'] / np.maximum(np.abs(10 ** dataset_test['output_log10_f_s_SI']), 1e-30))

        dataset_test['output_temperature_1000bar_K_pred_diff'] = np.abs(
            dataset_test['temperature_1000bar_K_pred'] - dataset_test['output_temperature_1000bar_K'])
        dataset_test['output_temperature_1000bar_K_pred_diff_ratio'] = (
            dataset_test['output_temperature_1000bar_K_pred_diff'] / dataset_test['output_temperature_1000bar_K'])

        logging.info("CV Test radius diff mean: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff'].std()))
        logging.info("CV Test entropy diff mean: " + str(
            dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff'].std()))
        logging.info("CV Test f_s diff mean: " + str(
            dataset_test.loc[:, 'output_f_s_SI_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_f_s_SI_pred_diff'].std()))
        logging.info("CV Test temperature diff mean: " + str(
            dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff'].std()))
        logging.info("CV Test radius diff ratio median: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("CV Test entropy diff ratio median (real): " + str(
            dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("CV Test f_s diff ratio median (real): " + str(
            dataset_test.loc[:, 'output_f_s_SI_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_f_s_SI_pred_diff_ratio'].std()))
        logging.info("CV Test temperature diff ratio median (real): " + str(
            dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff_ratio'].median()) + '+-' +
                     str(dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff_ratio'].std()))
        logging.info("CV Test radius diff ratio mean: " + str(
            dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_radius_Rearth_pred_diff_ratio'].std()))
        logging.info("CV Test entropy diff ratio mean (real): " + str(
            dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_entropy_1000bar_SI_pred_diff_ratio'].std()))
        logging.info("CV Test f_s diff ratio mean (real): " + str(
            dataset_test.loc[:, 'output_f_s_SI_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_f_s_SI_pred_diff_ratio'].std()))
        logging.info("CV Test temperature diff ratio mean (real): " + str(
            dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff_ratio'].mean()) + '+-' +
                     str(dataset_test.loc[:, 'output_temperature_1000bar_K_pred_diff_ratio'].std()))
        return dataset_test

    def predict_cv(self, df, model_dir, folds=5, batch_size=128):
        """
        Run all CV fold models on a DataFrame of input rows. Returns the DataFrame
        with added columns for ensemble median predictions and stds (real units).

        Parameters
        ----------
        df : pd.DataFrame  Must contain the 6 input columns (input_mass_Mearth, etc.)
        model_dir : str    Path to CV output (contains Haunter_model_0/, ...)
        folds : int        Number of CV folds
        batch_size : int

        Returns
        -------
        df : pd.DataFrame  Original df with added columns:
            radius_Rearth_pred, radius_Rearth_std
            entropy_1000bar_SI_pred, entropy_1000bar_SI_std
            f_s_SI_pred, f_s_SI_std
            temperature_1000bar_K_pred, temperature_1000bar_K_std
        """
        fold0_dir = f"{model_dir}/Haunter_model_0"
        dataset_training = pd.read_csv(f"{fold0_dir}/train_dataset.csv")
        dataset_validation = pd.read_csv(f"{fold0_dir}/validation_dataset.csv")
        dataset_standardization = pd.concat([dataset_training, dataset_validation])

        # Standardize inputs only (using training data stats)
        df_std = df.copy()
        input_cols = ['input_mass_Mearth', 'input_CMF', 'input_Zenv',
                      'input_Zwater_core', 'input_Tsurf_K', 'input_Psurf_bar']
        for col in input_cols:
            scaler = self.instance_scaler()
            scaler.fit(dataset_standardization[col])
            df_std.loc[:, col + '_norm'] = scaler.transform(df_std[col].to_numpy())
        # Generator needs output _norm columns present (values don't matter for prediction)
        for col in ['output_radius_Rearth', 'output_entropy_log10_1000bar_SI',
                     'output_log10_f_s_SI', 'output_log10_temperature_1000bar_K']:
            if col + '_norm' not in df_std.columns:
                df_std[col + '_norm'] = 0.0
        predict_generator = self.instance_generator(df_std, None, batch_size, self.input_size, None,
                                                    shuffle=False)

        all_preds = []
        for i in range(folds):
            model_path = f"{model_dir}/Haunter_model_{i}/model{self.MODEL_SAVE_EXTENSION}"
            logging.info(f"CV predict: loading fold {i} from {model_path}")
            self.load_model(model_path, compile=False)
            preds = self.model.predict(predict_generator, batch_size=batch_size, verbose=0)
            all_preds.append(preds)
            del self.model
            self.model = None

        all_preds = np.stack(all_preds, axis=0)  # (folds, n_samples, 4)
        pred_median = np.median(all_preds, axis=0)
        pred_std = all_preds.std(axis=0)

        # Destandardize using the same normalization
        output_stds = np.array([
            dataset_standardization['output_radius_Rearth'].std(),
            dataset_standardization['output_entropy_log10_1000bar_SI'].std(),
            dataset_standardization['output_log10_f_s_SI'].std(),
            dataset_standardization['output_log10_temperature_1000bar_K'].std(),
        ])
        output_means = np.array([
            dataset_standardization['output_radius_Rearth'].mean(),
            dataset_standardization['output_entropy_log10_1000bar_SI'].mean(),
            dataset_standardization['output_log10_f_s_SI'].mean(),
            dataset_standardization['output_log10_temperature_1000bar_K'].mean(),
        ])

        # Destandardize predictions
        pred_real = pred_median * output_stds + output_means  # (n_samples, 4)
        pred_real_std = pred_std * output_stds

        # Convert to real units: radius already real; entropy/f_s/temp from log10
        df['radius_Rearth_pred'] = pred_real[:, 0]
        df['radius_Rearth_std'] = pred_real_std[:, 0]

        # Exponentiate each fold's predictions then take std — exact, no approximation
        all_preds_real = all_preds * output_stds + output_means  # (folds, n, 4) in log10/real space
        all_preds_real_exp = 10 ** all_preds_real  # entropy, f_s, temp in real units, radius unchanged

        df['entropy_1000bar_SI_pred'] = np.median(all_preds_real_exp[:, :, 1], axis=0)
        df['entropy_1000bar_SI_std'] = all_preds_real_exp[:, :, 1].std(axis=0)
        df['f_s_SI_pred'] = np.median(all_preds_real_exp[:, :, 2], axis=0)
        df['f_s_SI_std'] = all_preds_real_exp[:, :, 2].std(axis=0)
        df['temperature_1000bar_K_pred'] = np.median(all_preds_real_exp[:, :, 3], axis=0)
        df['temperature_1000bar_K_std'] = all_preds_real_exp[:, :, 3].std(axis=0)

        logging.info(f"CV predict: {folds} folds ensemble on {len(df)} samples complete")
        return df

    def plot_hists(self, df, additional_columns=[]):
        columns = ['mass_Mearth_input', 'mass_Mearth_input_norm',
                   'CMF_input', 'CMF_input_norm',
                   'Zenv_input', 'Zenv_input_norm',
                   'Zwater_core_input', 'Zwater_core_input_norm',
                   'Tsurf_K_input', 'Tsurf_K_input_norm',
                   'Psurf_bar_input', 'Psurf_bar_input_norm',
                   'radius_Rearth_output', 'radius_Rearth_output_norm',
                   'output_log10_entropy_1000bar_SI', 'output_entropy_log10_1000bar_SI_norm',
                   'f_s_SI_output', 'f_s_SI_output_norm',
                   'log10_mass_integral_output', 'log10_mass_integral_output_norm'
                   ]
        fig, axs = plt.subplots(10, 2, figsize=(10, 30))
        for i, col in enumerate(columns):
            axs[i // 2][i % 2].hist(df[col].dropna(), bins=15, color='lightblue', edgecolor='black')
            axs[i // 2][i % 2].set_title(f"Distribution of {col}")
            axs[i // 2][i % 2].set_xlabel(col)
            axs[i // 2][i % 2].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        if len(additional_columns) > 0:
            fig, axs = plt.subplots(len(additional_columns) // 2, 2, figsize=(10, 30))
            for i, col in enumerate(additional_columns):
                axs[i // 2][i % 2].hist(df[col].dropna(), bins=15, color='lightblue', edgecolor='black')
                axs[i // 2][i % 2].set_title(f"Distribution of {col}")
                axs[i // 2][i % 2].set_xlabel(col)
                axs[i // 2][i % 2].set_ylabel('Frequency')
            plt.tight_layout()
            plt.show()


    def mse_radius(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        y_true_outputs = y_true_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        return tf.reduce_mean(tf.square(y_true_outputs[:, 0] - y_pred_outputs[:, 0]))

    def mse_entropy(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        y_true_outputs = y_true_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        return tf.reduce_mean(tf.square(y_true_outputs[:, 1] - y_pred_outputs[:, 1]))

    def mse_fs(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        y_true_outputs = y_true_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        return tf.reduce_mean(tf.square(y_true_outputs[:, 2] - y_pred_outputs[:, 2]))

    def mae_radius(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_stds + 1e-8) + self.output_means
        y_true_outputs = y_true_outputs * (self.output_stds + 1e-8) + self.output_means
        return tf.reduce_mean(tf.abs(y_true_outputs[:, 0] - y_pred_outputs[:, 0]))

    def mae_entropy(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_stds + 1e-8) + self.output_means
        y_true_outputs = y_true_outputs * (self.output_stds + 1e-8) + self.output_means
        return tf.reduce_mean(tf.abs(y_true_outputs[:, 1] - y_pred_outputs[:, 1]))

    def mae_fs(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_stds + 1e-8) + self.output_means
        y_true_outputs = y_true_outputs * (self.output_stds + 1e-8) + self.output_means
        return tf.reduce_mean(tf.abs(y_true_outputs[:, 2] - y_pred_outputs[:, 2]))

    def mae_temperature(self, y_true, y_pred):
        y_pred_outputs = y_pred
        y_true_outputs = y_true
        y_pred_outputs = y_pred_outputs * (self.output_stds + 1e-8) + self.output_means
        y_true_outputs = y_true_outputs * (self.output_stds + 1e-8) + self.output_means
        return tf.reduce_mean(tf.abs(y_true_outputs[:, 3] - y_pred_outputs[:, 3]))

    def combined_loss(self, y_true, y_pred):
        y_pred_outputs = y_pred[:, :3]
        y_pred_inputs = y_pred[:, 3:]
        y_true_outputs = y_true[:, :3]
        y_pred_inputs = y_pred_inputs * (self.input_maxs - self.input_mins + 1e-8) + self.input_mins
        y_pred_outputs = y_pred_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        y_true_outputs = y_true_outputs * (self.output_maxs - self.output_mins + 1e-8) + self.output_mins
        # Original inputs
        mass = y_pred_inputs[:, 0] * M_EARTH
        CMF = y_pred_inputs[:, 1]
        Zenv = y_pred_inputs[:, 2]
        Zwater = y_pred_inputs[:, 3]
        Tsurf = y_pred_inputs[:, 4]
        Psurf = y_pred_inputs[:, 5] * P_PASCAL
        # Predicted outputs
        radius_pred = y_pred_outputs[:, 0] * R_EARTH
        entropy_pred = y_pred_outputs[:, 1]
        f_s_pred = y_pred_outputs[:, 2]
        # Original outputs
        radius_true = y_true_outputs[:, 0] * R_EARTH
        entropy_true = y_true_outputs[:, 1]
        f_s_true = y_true_outputs[:, 2]
        # ---------- Average density by mass/radius ----------
        rho_mean_pred = mass / (4.0 / 3.0 * tf.constant(PI) * tf.pow(radius_pred, 3))
        rho_mean_true = mass / (4.0 / 3.0 * tf.constant(PI) * tf.pow(radius_true, 3))
        # ---------- Composition density ----------
        rho_comp = (
            CMF * RHO_CORE +
            Zwater * RHO_WATER +
            Zenv * RHO_GAS +
            (1.0 - CMF - Zwater - Zenv) * RHO_MANTLE
        )
        # ---------- Surface gravity ----------
        g_surface_pred = G * mass / tf.square(radius_pred)
        g_surface_true = G * mass / tf.square(radius_pred)
        # ---------- Physical equations (residuals) ----------
        # Eq1: dP/dr ≈ -rho * g
        dP_dr_estimate_pred = -rho_mean_pred * g_surface_pred
        dP_dr_estimate_true = -rho_mean_true * g_surface_true
        loss_dpdr = tf.reduce_mean(tf.abs((Psurf / radius_pred - dP_dr_estimate_pred) - (Psurf / radius_true - dP_dr_estimate_true)))
        # Eq2: dg/dr ≈ -4πGρ - 2Gm/r³
        dg_dr_estimate_pred = -4 * tf.constant(PI) * G * rho_mean_pred - 2 * G * mass / tf.pow(radius_pred, 3)
        dg_dr_estimate_true = -4 * tf.constant(PI) * G * rho_mean_true - 2 * G * mass / tf.pow(radius_pred, 3)
        loss_dgdr = tf.reduce_mean(tf.abs(dg_dr_estimate_pred - dg_dr_estimate_true))
        # Eq3: dT/dr ≈ -g γT/φ, using γ ≈ entropy * escalar
        gamma_pred = entropy_pred * GAMMA_SCALING
        gamma_true = entropy_true * GAMMA_SCALING
        phi_pred = Psurf / (rho_mean_pred + 1e-6)
        phi_true = Psurf / (rho_mean_true + 1e-6)
        dT_dr_estimate_pred = -g_surface_pred * gamma_pred * Tsurf / (phi_pred + 1e-6)
        dT_dr_estimate_true = -g_surface_true * gamma_true * Tsurf / (phi_true + 1e-6)
        loss_dTdr = tf.reduce_mean(tf.abs(dT_dr_estimate_pred - dT_dr_estimate_true))
        # Eq6: dm/dr ≈ 4πr²ρ
        dm_dr_estimate_pred = 4.0 * tf.constant(PI) * tf.square(radius_pred) * rho_mean_pred
        dm_dr_estimate_true = 4.0 * tf.constant(PI) * tf.square(radius_true) * rho_mean_true
        loss_dmdr = tf.reduce_mean(tf.abs((mass / radius_pred - dm_dr_estimate_pred) - (mass / radius_true - dm_dr_estimate_true)))
        # Densities consistency (estimated by percents of composition vs mass/radius densities)
        loss_rho_match = tf.reduce_mean(tf.abs((rho_mean_pred - rho_comp) - (rho_mean_true - rho_comp)))


        # Additional f_s losses
        f_s_true_abs = tf.abs(f_s_true)
        f_s_pred_abs = tf.abs(f_s_pred)
        f_s_true_log = np.log10(f_s_true_abs)  # this is option 1
        f_s_pred_log = np.log10(f_s_pred_abs)  # this is option 1
        f_s_true_mass_factor = f_s_true_abs / (mass)
        f_s_pred_mass_factor = f_s_pred_abs / (mass)
        log10_mass_integral_true = np.log10(f_s_true_mass_factor)  # this is option 2
        log10_mass_integral_pred = np.log10(f_s_pred_mass_factor)  # this is option 2
        log10_mass_integral_loss = tf.reduce_mean(tf.square(log10_mass_integral_true - log10_mass_integral_pred))


        # MSE
        radius_mse_outputs = tf.reduce_mean(tf.square(radius_true - radius_pred))
        entropy_mse_outputs = tf.reduce_mean(tf.square(entropy_true - entropy_pred))
        f_s_mae_outputs = tf.reduce_mean(tf.abs(f_s_true - f_s_pred))
        mse = tf.reduce_mean(mean_squared_error(y_true, y_pred))
        total_loss = mse + log10_mass_integral_loss
        # total_loss = (
        #     total_loss +
        #     0.2 * loss_dpdr +
        #     0.2 * loss_dgdr +
        #     0.2 * loss_dTdr +
        #     0.2 * loss_dmdr +
        #     0.2 * loss_rho_match
        # )
        return total_loss

class StandardizeMinMaxScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.mean_ = np.nanmean(data, axis=0)
        self.std_ = np.nanstd(data, axis=0)
        standardized = (data - self.mean_) / (self.std_ + 1e-8)
        self.min_ = np.nanmin(standardized, axis=0)
        self.max_ = np.nanmax(standardized, axis=0)

    def transform(self, data):
        standardized = (data - self.mean_) / (self.std_ + 1e-8)
        scaled = (standardized - self.min_) / (self.max_ - self.min_ + 1e-8)
        indices = np.argwhere(scaled == 0.0).flatten()
        scaled[indices] = 1e-8
        indices = np.argwhere(scaled == 1.0).flatten()
        scaled[indices] = 1 - 1e-8
        return scaled

    def inverse_transform(self, data_scaled):
        standardized = data_scaled * (self.max_ - self.min_ + 1e-8) + self.min_
        original = standardized * (self.std_ + 1e-8) + self.mean_
        return original

class MinMaxScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.min_ = np.nanmin(data, axis=0)
        self.max_ = np.nanmax(data, axis=0)

    def transform(self, data):
        scaled = (data - self.min_) / (self.max_ - self.min_ + 1e-8)
        indices = np.argwhere(scaled == 0.0).flatten()
        scaled[indices] = 1e-8
        indices = np.argwhere(scaled == 1.0).flatten()
        scaled[indices] = 1 - 1e-8
        return scaled

    def inverse_transform(self, data_scaled):
        return data_scaled * (self.max_ - self.min_ + 1e-8) + self.min_

class StandardizeScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        self.mean_ = np.nanmean(data, axis=0)
        self.std_ = np.nanstd(data, axis=0)

    def transform(self, data):
        return (data - self.mean_) / (self.std_ + 1e-8)

    def inverse_transform(self, data_scaled):
        return data_scaled * (self.std_ + 1e-8) + self.mean_


