import logging

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model

from exoml.haunter.haunter_generator import HaunterModelGenerator, INPUT_FEATURES, OUTPUT_TARGETS, NORM
from exoml.ml.model.base_model import BaseModel, HyperParams

RAW_INPUTS = ['input_mass_Mearth', 'input_CMF', 'input_Zenv', 'input_Zwater_core', 'input_Tsurf_K', 'input_Psurf_bar']
# name, linear column, the way it is transformed to the learned target
OUTPUT_NAMES = ['radius', 'entropy', 'f_s', 'temperature']
LINEAR_OUTPUTS = {'radius': 'output_radius_Rearth', 'entropy': 'output_entropy_1000bar_SI',
                  'temperature': 'output_temperature_1000bar_K'}


class Haunter(BaseModel):
    """
    Predicts the radius, the entropy, the f_s integral and the temperature at 1000 bar of a planetary interior from its
    mass, core mass fraction, envelope and water metallicity, surface temperature and surface pressure.

    The inputs that span orders of magnitude (mass, pressure) and the outputs that do (radius, entropy, temperature)
    are learned in log10, so that the relative error is what the network minimizes. The simulations are deterministic,
    so the network is a plain residual perceptron without dropout, noise or batch normalization.
    """
    def __init__(self, hyperparameters: HyperParams, name='Haunter', input_size=len(INPUT_FEATURES), width=256, blocks=3):
        super().__init__(name, input_size, None, None, hyperparameters)
        self.width = width
        self.blocks = blocks
        self.output_means = None
        self.output_stds = None

    @staticmethod
    def add_derived_columns(dataset):
        """The columns the network learns from the raw columns of the simulation (log10 of the wide ones)."""
        dataset = dataset.copy()
        dataset['input_log10_mass_Mearth'] = np.log10(dataset['input_mass_Mearth'])
        dataset['input_log10_Psurf_bar'] = np.log10(dataset['input_Psurf_bar'])
        dataset['input_log10_distance_to_1000bar'] = np.log10(1000. - dataset['input_Psurf_bar'] + 0.01)
        dataset['output_log10_radius_Rearth'] = np.log10(dataset['output_radius_Rearth'])
        dataset['output_entropy_log10_1000bar_SI'] = np.log10(dataset['output_entropy_1000bar_SI'])
        dataset['output_log10_temperature_1000bar_K'] = np.log10(dataset['output_temperature_1000bar_K'])
        return dataset

    @staticmethod
    def physical_rows(dataset):
        """Only the converged, physical simulations are kept."""
        dataset = dataset.loc[(dataset['flag_physical'] == True) & (dataset['flag_convergence'] == 0)]
        return dataset.dropna()

    @staticmethod
    def load_data(dir):
        dataset = pd.read_csv(f"{dir}/models_for_ML_training_noduplicates.csv", sep=',')
        return Haunter.add_derived_columns(Haunter.physical_rows(dataset))

    def build(self):
        input = keras.Input(shape=(self.input_size,))
        nn = keras.layers.Dense(self.width, activation='silu')(input)
        for _ in range(self.blocks):
            branch = keras.layers.Dense(self.width, activation='silu')(nn)
            branch = keras.layers.Dense(self.width)(branch)
            nn = keras.layers.Activation('silu')(keras.layers.Add()([nn, branch]))
        nn = keras.layers.Dense(len(OUTPUT_TARGETS), activation="linear")(nn)
        self.set_model(Model(inputs=input, outputs=nn))
        return self

    def load_training_set(self, **kwargs):
        dataset = Haunter.load_data(kwargs.get('training_dir'))
        self.output_means = tf.constant(dataset[OUTPUT_TARGETS].mean().to_numpy(), dtype=tf.float32)
        self.output_stds = tf.constant(dataset[OUTPUT_TARGETS].std().to_numpy(), dtype=tf.float32)
        return self.add_standardized_columns(dataset, dataset)

    def instance_scaler(self):
        return StandardizeScaler()

    def add_standardized_columns(self, dataset, training_dataset):
        """Adds the ``_norm`` columns, standardized with the mean and deviation of the training dataset."""
        for column in INPUT_FEATURES + OUTPUT_TARGETS:
            if column not in dataset:
                continue
            scaler = self.instance_scaler()
            scaler.fit(training_dataset.loc[:, column])
            dataset.loc[:, column + NORM] = scaler.transform(dataset.loc[:, column].to_numpy())
        return dataset

    def add_destandardized_columns(self, dataset, training_dataset, predictions):
        """Adds the ``_pred`` columns in the learned space and the physical value of every output."""
        for i, column in enumerate(OUTPUT_TARGETS):
            scaler = self.instance_scaler()
            scaler.fit(training_dataset.loc[:, column])
            dataset.loc[:, column + '_pred'] = scaler.inverse_transform(predictions[:, i])
        return self.add_physical_predictions(dataset)

    @staticmethod
    def add_physical_predictions(dataset):
        """The predicted radius, entropy and temperature in their physical units, from the learned log10 values."""
        dataset['output_radius_Rearth_pred'] = 10 ** dataset['output_log10_radius_Rearth_pred']
        dataset['output_entropy_1000bar_SI_pred'] = 10 ** dataset['output_entropy_log10_1000bar_SI_pred']
        dataset['output_temperature_1000bar_K_pred'] = 10 ** dataset['output_log10_temperature_1000bar_K_pred']
        return dataset

    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon=1e-7, shuffle=True):
        return HaunterModelGenerator(dataset, batch_size, input_sizes, zero_epsilon, shuffle=shuffle)

    def instance_loss_accuracy(self):
        # the targets are standardized: a quadratic loss close to zero and a linear one for the few hard models
        return keras.losses.Huber(delta=0.1), None

    def instance_metrics(self):
        return [self.mae_radius, self.mae_entropy, self.mae_fs, self.mae_temperature]

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        return []

    def load_model(self, dir, custom_objects={}, compile=False):
        for metric in (self.mae_radius, self.mae_entropy, self.mae_fs, self.mae_temperature):
            custom_objects[metric.__name__] = metric
        logging.info("Loading model from %s", dir)
        self.set_model(tf.keras.models.load_model(dir, compile=compile, custom_objects=custom_objects))
        return self

    def _mae(self, y_true, y_pred, index):
        """The mean absolute error of one output in its learned space (dex for the log10 outputs)."""
        scale = self.output_stds[index] + 1e-8
        return tf.reduce_mean(tf.abs(y_true[:, index] - y_pred[:, index])) * scale

    def mae_radius(self, y_true, y_pred):
        return self._mae(y_true, y_pred, 0)

    def mae_entropy(self, y_true, y_pred):
        return self._mae(y_true, y_pred, 1)

    def mae_fs(self, y_true, y_pred):
        return self._mae(y_true, y_pred, 2)

    def mae_temperature(self, y_true, y_pred):
        return self._mae(y_true, y_pred, 3)

    @staticmethod
    def evaluate_predictions(dataset):
        """
        The error of the predictions of every output, in the learned space (``*_pred_diff``, dex for the log10 ones)
        and relative to the physical value (``*_pred_diff_ratio``). f_s is only known as log10.
        Returns a table with the mean absolute error, and the median, mean and 95th percentile of the relative error.
        """
        rows = {}
        for name, column in {**LINEAR_OUTPUTS, 'f_s': 'output_log10_f_s_SI'}.items():
            truth, pred = dataset[column], dataset[column + '_pred']
            dataset[column + '_pred_diff'] = (pred - truth).abs()
            if name == 'f_s':
                truth, pred = 10 ** truth, 10 ** pred
            dataset[column + '_pred_diff_ratio'] = ((pred - truth) / truth).abs()
            ratio = dataset[column + '_pred_diff_ratio']
            rows[name] = {'median_rel_%': 100 * ratio.median(), 'mean_rel_%': 100 * ratio.mean(),
                          'p95_rel_%': 100 * ratio.quantile(.95)}
        for name, target in zip(OUTPUT_NAMES, OUTPUT_TARGETS):
            dataset[target + '_pred_diff'] = (dataset[target + '_pred'] - dataset[target]).abs()
            rows[name]['mae_learned_space'] = float(dataset[target + '_pred_diff'].mean())
        table = pd.DataFrame(rows).T
        logging.info("Relative and absolute errors:\n%s", table.round(4).to_string())
        return table

    def _predict_dataset(self, dataset, standardization, batch_size):
        dataset = self.add_standardized_columns(dataset, standardization)
        generator = self.instance_generator(dataset, None, batch_size, self.input_size, None, shuffle=False)
        predictions = self.model.predict(generator, batch_size=batch_size)
        dataset = self.add_destandardized_columns(dataset, standardization, predictions)
        self.evaluate_predictions(dataset)
        return dataset

    def predict_validation(self, model_dir, dir, batch_size=128):
        dataset_training = pd.read_csv(f"{model_dir}/train_dataset.csv")
        dataset_validation = pd.read_csv(f"{model_dir}/validation_dataset.csv")
        return self._predict_dataset(dataset_validation, pd.concat([dataset_training, dataset_validation]), batch_size)

    def predict_test(self, model_dir, test_csv, batch_size=128):
        dataset_training = pd.read_csv(f"{model_dir}/train_dataset.csv")
        dataset_validation = pd.read_csv(f"{model_dir}/validation_dataset.csv")
        dataset_test = Haunter.add_derived_columns(Haunter.physical_rows(pd.read_csv(test_csv)))
        return self._predict_dataset(dataset_test, pd.concat([dataset_training, dataset_validation]), batch_size)

    def predict_test_cv(self, model_dir, test_csv, folds=5, batch_size=128):
        dataset_test = Haunter.add_derived_columns(Haunter.physical_rows(pd.read_csv(test_csv)))
        dataset_test = self.predict_cv(dataset_test, model_dir, folds=folds, batch_size=batch_size)
        self.add_physical_predictions(dataset_test)
        self.evaluate_predictions(dataset_test)
        return dataset_test

    def predict_cv(self, df, model_dir, folds=5, batch_size=128):
        """
        Runs the model of every fold on a dataframe with the six raw input columns and adds the median of the folds
        (``<target>_pred`` in the learned space) and the dispersion between folds (``<target>_pred_std``). The
        dispersion between folds is the spread of the ensemble, not a calibrated error: compare it to the real error
        before using it as an uncertainty.
        """
        df = df.copy()
        features = Haunter.add_derived_columns(
            df.assign(output_radius_Rearth=1., output_entropy_1000bar_SI=1., output_temperature_1000bar_K=1.))
        all_preds = []
        for i in range(folds):
            fold_dir = f"{model_dir}/Haunter_model_{i}"
            # every fold was trained with its own standardization
            standardization = pd.concat([pd.read_csv(f"{fold_dir}/train_dataset.csv"),
                                         pd.read_csv(f"{fold_dir}/validation_dataset.csv")])
            batch = self.add_standardized_columns(features.copy(), standardization)
            for target in OUTPUT_TARGETS:
                batch[target + NORM] = 0.0
            generator = self.instance_generator(batch, None, batch_size, self.input_size, None, shuffle=False)
            model_path = f"{fold_dir}/model{self.MODEL_SAVE_EXTENSION}"
            logging.info(f"CV predict: loading fold {i} from {model_path}")
            self.load_model(model_path, compile=False)
            preds = self.model.predict(generator, batch_size=batch_size, verbose=0)
            scaler_stats = standardization[OUTPUT_TARGETS]
            all_preds.append(preds * scaler_stats.std().to_numpy() + scaler_stats.mean().to_numpy())
            del self.model
            self.model = None
        all_preds = np.stack(all_preds, axis=0)  # (folds, n_samples, 4) in the learned space
        for i, target in enumerate(OUTPUT_TARGETS):
            df[target + '_pred'] = np.median(all_preds[:, :, i], axis=0)
            df[target + '_pred_std'] = all_preds[:, :, i].std(axis=0)
        logging.info(f"CV predict: {folds} folds ensemble on {len(df)} samples complete")
        return df


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
