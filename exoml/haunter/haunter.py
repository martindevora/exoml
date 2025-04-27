import logging
import tensorflow as tf
import numpy as np

import keras
import pandas as pd
from keras import Model
from keras.layers import LeakyReLU
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import RootMeanSquaredError
from keras_core.src.metrics import R2Score

from exoml.haunter.haunter_generator import HaunterModelGenerator
from exoml.ml.log.with_logging import WithLogging
from exoml.ml.loss.unsupervised import intransit_weighted_mean_squared_error
from exoml.ml.metrics.fixed_mean import FixedMean

from exoml.ml.model.base_model import BaseModel, HyperParams

def mse_radius(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))

def mse_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))

def mse_fs(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))

def rmse_radius(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0])))

def rmse_entropy(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1])))

def rmse_fs(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2])))

def mae_radius(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

def mae_entropy(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

def mae_fs(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 2] - y_pred[:, 2]))

def r2_radius(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true[:, 0] - y_pred[:, 0]))
    ss_tot = tf.reduce_sum(tf.square(y_true[:, 0] - tf.reduce_mean(y_true[:, 0])))
    return 1 - ss_res / (ss_tot + 1e-7)  # Add epsilon to avoid division by zero

def r2_entropy(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true[:, 1] - y_pred[:, 1]))
    ss_tot = tf.reduce_sum(tf.square(y_true[:, 1] - tf.reduce_mean(y_true[:, 1])))
    return 1 - ss_res / (ss_tot + 1e-7)

def r2_fs(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true[:, 2] - y_pred[:, 2]))
    ss_tot = tf.reduce_sum(tf.square(y_true[:, 2] - tf.reduce_mean(y_true[:, 2])))
    return 1 - ss_res / (ss_tot + 1e-7)


class StandardizeMinMaxScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        standardized = (data - self.mean_) / (self.std_ + 1e-8)
        self.min_ = np.min(standardized, axis=0)
        self.max_ = np.max(standardized, axis=0)

    def transform(self, data):
        standardized = (data - self.mean_) / (self.std_ + 1e-8)
        scaled = (standardized - self.min_) / (self.max_ - self.min_ + 1e-8)
        return scaled

    def inverse_transform(self, data_scaled):
        standardized = data_scaled * (self.max_ - self.min_ + 1e-8) + self.min_
        original = standardized * (self.std_ + 1e-8) + self.mean_
        return original


class Haunter(BaseModel):
    def __init__(self, hyperparameters: HyperParams, name='Haunter', input_size=6):
        super().__init__(name, input_size, None, None, hyperparameters)

    def build(self):
        input = keras.Input(shape=(self.input_size))
        nn = keras.layers.Dense(32, activation=LeakyReLU(0.01))(input)
        nn = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(nn)
        nn = keras.layers.Dense(128, activation=LeakyReLU(0.01))(nn)
        nn = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(nn)
        nn = keras.layers.Dense(32, activation=LeakyReLU(0.01))(nn)
        nn = keras.layers.Dropout(rate=self.hyperparams.dropout_rate)(nn)
        nn = keras.layers.Dense(3, activation="linear")(nn)
        self.set_model(Model(inputs=input, outputs=nn))
        return self

    def load_training_set(self, **kwargs):
        training_dir = kwargs.get('training_dir')
        dataset = pd.read_csv(f"{training_dir}/table.dat", sep=' ')
        dataset = dataset.loc[dataset['physical?'] == True]
        dataset = self.add_standardized_columns(dataset, dataset)
        return dataset

    def add_standardized_columns(self, dataset, training_dataset):
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'mass_Mearth_input'])
        dataset.loc[:, 'mass_Mearth_input_norm'] = scaler.transform(dataset.loc[:, 'mass_Mearth_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'CMF_input'])
        dataset.loc[:, 'CMF_input_norm'] = scaler.transform(dataset.loc[:, 'CMF_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'Zenv_input'])
        dataset.loc[:, 'Zenv_input_norm'] = scaler.transform(dataset.loc[:, 'Zenv_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'Zwater_core_input'])
        dataset.loc[:, 'Zwater_core_input_norm'] = scaler.transform(dataset.loc[:, 'Zwater_core_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'Tsurf_K_input'])
        dataset.loc[:, 'Tsurf_K_input_norm'] = scaler.transform(dataset.loc[:, 'Tsurf_K_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'Psurf_bar_input'])
        dataset.loc[:, 'Psurf_bar_input_norm'] = scaler.transform(dataset.loc[:, 'Psurf_bar_input'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'radius_Rearth_output'])
        dataset.loc[:, 'radius_Rearth_output_norm'] = scaler.transform(dataset.loc[:, 'radius_Rearth_output'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'entropy_SI_output'])
        dataset.loc[:, 'entropy_SI_output_norm'] = scaler.transform(dataset.loc[:, 'entropy_SI_output'])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'f_s_SI_output'])
        dataset.loc[:, 'f_s_SI_output_norm'] = scaler.transform(dataset.loc[:, 'f_s_SI_output'])
        return dataset

    def add_destandardized_columns(self, dataset, training_dataset, predictions):
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'radius_Rearth_output'])
        dataset.loc[:, 'radius_Rearth_output_pred'] = scaler.transform(predictions[:, 0])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'entropy_SI_output'])
        dataset.loc[:, 'entropy_SI_output_pred'] = scaler.transform(predictions[:, 1])
        scaler = StandardizeMinMaxScaler()
        scaler.fit(training_dataset.loc[:, 'f_s_SI_output'])
        dataset.loc[:, 'f_s_SI_output_pred'] = scaler.transform(predictions[:, 2])
        return dataset

    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon=1e-7, shuffle=True):
        return HaunterModelGenerator(dataset, batch_size, input_sizes, zero_epsilon, shuffle=shuffle)

    def instance_loss_accuracy(self):
        #loss = tf.keras.losses.MeanAbsolutePercentageError(reduction=losses_utils.ReductionV2.AUTO)
        loss = MeanSquaredError()
        accuracy = FixedMean()
        return loss, accuracy

    def instance_metrics(self):
        return [mse_radius, mse_entropy, mse_fs, rmse_radius, rmse_entropy, rmse_fs,
                mae_radius, mae_entropy, mae_fs, r2_radius, r2_entropy, r2_fs]

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        return []

    def load_model(self, dir, custom_objects={}, compile=False):
        custom_objects['mse_radius'] = mse_radius
        custom_objects['mse_entropy'] = mse_entropy
        custom_objects['mse_fs'] = mse_fs
        custom_objects['rmse_radius'] = rmse_radius
        custom_objects['rmse_entropy'] = rmse_entropy
        custom_objects['rmse_fs'] = rmse_fs
        custom_objects['mae_radius'] = mae_radius
        custom_objects['mae_entropy'] = mae_entropy
        custom_objects['mae_fs'] = mae_fs
        custom_objects['r2_radius'] = r2_radius
        custom_objects['r2_entropy'] = r2_entropy
        custom_objects['r2_fs'] = r2_fs
        logging.info("Loading model from %s", dir)
        self.set_model(tf.keras.models.load_model(dir, compile=compile, custom_objects=custom_objects))
        return self

    def predict_non_physical(self, dir, batch_size=128):
        dataset = pd.read_csv(f"{dir}/table.dat", sep=' ')
        dataset_training = dataset.loc[dataset['physical?'] == True]
        dataset_testing = dataset.loc[dataset['physical?'] == False]
        dataset_testing = self.add_standardized_columns(dataset_testing, dataset_training)
        predict_batch_generator = self.instance_generator(dataset_testing, None, batch_size, self.input_size, None)
        predictions = self.model.predict(predict_batch_generator, batch_size=batch_size, workers=1)
        dataset_testing['radius_Rearth_output_norm_pred'] = predictions[:, 0]
        dataset_testing['entropy_SI_output_norm_pred'] = predictions[:, 1]
        dataset_testing['f_s_SI_output_norm_pred'] = predictions[:, 2]
        dataset_testing = self.add_destandardized_columns(dataset_testing, dataset_training, predictions)
        dataset_testing['radius_Rearth_output_pred_diff'] = np.abs(dataset_testing['radius_Rearth_output_pred'] - dataset_testing['radius_Rearth_output'])
        dataset_testing['entropy_SI_output_pred_diff'] = np.abs(dataset_testing['entropy_SI_output_pred'] - dataset_testing['entropy_SI_output'])
        dataset_testing['f_s_SI_output_pred_diff'] = np.abs(dataset_testing['f_s_SI_output_pred'] - dataset_testing['f_s_SI_output'])
        dataset_testing['radius_Rearth_output_pred_diff_ratio'] = np.abs(dataset_testing['radius_Rearth_output_pred'] - dataset_testing['radius_Rearth_output']) / dataset_testing['radius_Rearth_output']
        dataset_testing['entropy_SI_output_pred_diff_ratio'] = np.abs(dataset_testing['entropy_SI_output_pred'] - dataset_testing['entropy_SI_output']) / dataset_testing['entropy_SI_output']
        dataset_testing['f_s_SI_output_pred_diff_ratio'] = np.abs(dataset_testing['f_s_SI_output_pred'] - dataset_testing['f_s_SI_output']) / dataset_testing['f_s_SI_output']
        results_df = dataset_testing[['radius_Rearth_output_pred', 'radius_Rearth_output', 'radius_Rearth_output_pred_diff', 'radius_Rearth_output_pred_diff_ratio', 'f_s_SI_output_pred', 'f_s_SI_output', 'f_s_SI_output_pred_diff', 'f_s_SI_output_pred_diff_ratio', 'entropy_SI_output_pred', 'entropy_SI_output', 'entropy_SI_output_pred_diff', 'entropy_SI_output_pred_diff_ratio']]

    def predict_validation(self, model_dir, dir, batch_size=128):
        dataset = pd.read_csv(f"{dir}/table.dat", sep=' ')
        dataset_training = dataset.loc[dataset['physical?'] == True]
        dataset_validation = pd.read_csv(f"{model_dir}/validation_dataset.csv")
        dataset_training, means_and_stds = self.add_standardized_columns(dataset_validation, dataset_training)
        predict_batch_generator = self.instance_generator(dataset_validation, None, batch_size, self.input_size, None)
        predictions = self.model.predict(predict_batch_generator, batch_size=batch_size, workers=1)
        dataset_validation['radius_Rearth_output_norm_pred'] = predictions[:, 0]
        dataset_validation['entropy_SI_output_norm_pred'] = predictions[:, 1]
        dataset_validation['f_s_SI_output_norm_pred'] = predictions[:, 2]
        dataset_validation = self.add_destandardized_columns(dataset_validation, dataset_training, predictions)
        dataset_validation['radius_Rearth_output_pred_diff'] = np.abs(dataset_validation['radius_Rearth_output_pred'] - dataset_validation['radius_Rearth_output'])
        dataset_validation['entropy_SI_output_pred_diff'] = np.abs(dataset_validation['entropy_SI_output_pred'] - dataset_validation['entropy_SI_output'])
        dataset_validation['f_s_SI_output_pred_diff'] = np.abs(dataset_validation['f_s_SI_output_pred'] - dataset_validation['f_s_SI_output'])
        dataset_validation['radius_Rearth_output_pred_diff_ratio'] = np.abs(dataset_validation['radius_Rearth_output_pred'] - dataset_validation['radius_Rearth_output']) / dataset_validation['radius_Rearth_output']
        dataset_validation['entropy_SI_output_pred_diff_ratio'] = np.abs(dataset_validation['entropy_SI_output_pred'] - dataset_validation['entropy_SI_output']) / dataset_validation['entropy_SI_output']
        dataset_validation['f_s_SI_output_pred_diff_ratio'] = np.abs(dataset_validation['f_s_SI_output_pred'] - dataset_validation['f_s_SI_output']) / dataset_validation['f_s_SI_output']
        results_df = dataset_validation[['radius_Rearth_output_pred', 'radius_Rearth_output',
                                         'radius_Rearth_output_pred_diff', 'radius_Rearth_output_pred_diff_ratio',
                                         'f_s_SI_output_pred', 'f_s_SI_output', 'f_s_SI_output_pred_diff',
                                         'f_s_SI_output_pred_diff_ratio', 'entropy_SI_output_pred', 'entropy_SI_output',
                                         'entropy_SI_output_pred_diff', 'entropy_SI_output_pred_diff_ratio']]
