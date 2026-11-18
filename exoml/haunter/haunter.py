import logging

import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from keras import Model
from keras.layers import LeakyReLU

from exoml.haunter.haunter_generator import HaunterModelGenerator

from exoml.ml.model.base_model import BaseModel, HyperParams



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
        loss = tf.keras.losses.MeanSquaredError()
        return loss, None

    def instance_metrics(self):
        return [self.mae_radius, self.mae_entropy, self.mae_fs, self.mae_temperature]

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        return []

    def load_model(self, dir, custom_objects={}, compile=False):
        custom_objects['LeakyReLU'] = LeakyReLU
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
