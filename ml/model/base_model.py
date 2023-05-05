import dataclasses
import json
import logging
import os
from copy import deepcopy
from typing import List, Dict, Union, Any, Optional, Literal

import numpy as np
import tensorflow as tf
import pandas as pd
import ruamel.yaml
from abc import abstractmethod

from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle

from ml.callback.basemodel_aware_callback import MetricsPlotCallback
from ml.callback.batch_aware_csv_logger import BatchAwareCsvLogger
from ml.callback.early_stopping import ExoMlEarlyStopping
from ml.callback.get_weights import GetWeights
from ml.callback.training_data_aware import ValidationDataAwareCallback, ModelDirDataAwareCallback
from ml.learning_rate.schedulers import ExponentialRescaleDecay
from ml.log.get_weights_logger import ModelWeightsLogger
from ml.log.with_logging import WithLogging
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule

from ml.metrics.auc import precision_at_k, mean_positive_value, mean_true_positive_value


@dataclasses.dataclass
class HyperParams:
    batch_size: int = 20
    epochs: int = 20
    initial_learning_rate: float = 0.01,
    dataset_iterations_per_epoch: float = 1
    train_percent: float = 0.8
    validation_percent: float = 0.1
    training_set_limit: Optional[int] = None
    dry_run: bool = True
    zero_epsilon = 1e-7
    gradient_clip_norm: Optional[float] = 0.01
    gradient_clip_value: Optional[float] = 0.5
    cores: int = 0
    metrics: List = dataclasses.field(default_factory=lambda: [])
    callbacks: List = dataclasses.field(default_factory=lambda: [])
    learning_rate_decay: float = 0.98
    learning_rate_schedule: Optional[LearningRateSchedule] = None
    balance_class_id: Optional[str] = None
    balance_class_sampling: Optional[List] = None
    class_loss_weights: Optional[Union[Dict, Literal['auto']]] = None
    early_stopping_delta: float = 0
    early_stopping_patience: int = 0
    custom_loss: Optional[Any] = None
    run_eagerly: bool = False
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    dropout_rate: float = 0.1
    white_noise_std: Optional[float] = 0.0


@dataclasses.dataclass
class CategoricalPredictionSetStats:
    """
    Statistics of categorical predictions.
    """
    tp: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    mismatches: Optional[Dict] = None
    mismatches_by_type: Optional[Dict] = None



class BaseModel(WithLogging):
    """
    Base class providing standard methods for model building.
    """

    model = None

    def __init__(self, name, input_size, class_ids, type_to_label, hyperparams) -> None:
        super().__init__()
        self.name = name
        self.input_size = input_size
        if class_ids is not None:
            for class_id in class_ids.keys():
                class_ids[class_id] = class_ids[class_id] if isinstance(class_ids[class_id], (list, np.ndarray)) \
                    else [class_ids[class_id]]
        self.class_ids = class_ids
        self.type_to_label = type_to_label
        self.hyperparams = hyperparams

    @abstractmethod
    def build(self, **kwargs):
        """
        Should be used to create all the model layers and related capabilities to be used in the training and subsequent
        tasks.
        :param kwargs:
        """
        pass

    @abstractmethod
    def load_training_set(self, **kwargs):
        """
        Loads the training set from its source
        :param kwargs: the case-specific params
        :return: either a list or a Pandas Dataframe
        """
        pass

    @abstractmethod
    def instance_generator(self, dataset, dir, batch_size, input_sizes, type_to_label, zero_epsilon):
        pass

    @abstractmethod
    def instance_loss_accuracy(self):
        pass

    @abstractmethod
    def instance_metrics(self):
        pass

    @abstractmethod
    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        pass

    def balance_dataset_with_sampling(self, training_set, classes_sampling):
        classes_counts = [len(training_set[training_set['type'].isin(labels)]) for labels in self.class_ids.values()]
        dropped_samples = training_set.iloc[:0, :].copy()
        for id, sampling in enumerate(classes_sampling):
            labels = self.class_ids[id]
            class_count = classes_counts[id]
            class_expected_size = class_count * sampling
            if sampling > 1:
                rows_for_type = training_set[training_set['type'].isin(labels)]
                sampling = sampling - 1
                training_set = [training_set, rows_for_type * np.floor(sampling).astype(int)]
                training_set = pd.concat(training_set)
                class_expected_size_remainder = np.round(np.mod(sampling, 1) * class_count).astype(int)
                training_set = [training_set, training_set[training_set['type'].isin(labels)].iloc[0:class_expected_size_remainder]]
                training_set = pd.concat(training_set)
            elif sampling < 1:
                class_expected_size = np.round(class_expected_size).astype(int)
                samples_to_keep = training_set[training_set['type'].isin(labels)].iloc[0:class_expected_size]
                samples_to_remove = training_set[training_set['type'].isin(labels)].iloc[class_expected_size:]
                dropped_samples = dropped_samples.append([samples_to_remove])
                training_set.drop(training_set[training_set['type'].isin(labels)].index, inplace=True)
                training_set = [training_set, samples_to_keep]
                training_set = pd.concat(training_set)
        return training_set, dropped_samples

    def balance_dataset_from_class(self, training_set, class_id):
        assert class_id in self.class_ids.keys()
        if not isinstance(training_set, pd.DataFrame):
            logging.warning("Cannot balance dataset because the input is not a dataframe")
            return training_set
        classes_counts = [len(training_set[training_set['type'].isin(labels)]) for labels in self.class_ids.values()]
        classes_sampling = [1 if id == class_id else classes_counts[class_id] / count
                            for id, count in enumerate(classes_counts)]
        return self.balance_dataset_with_sampling(training_set, classes_sampling)

    def compile(self, optimizer, loss, metrics=None, run_eagerly=False, tuner=False):
        """
        Compiles and prepares the model for training
        :param optimizer: the optimizer to be used
        :param loss: the loss to be used
        :param metrics: the metrics to be used
        :return: the object itself
        """
        if metrics is None:
            metrics = []
        metrics_str = [str(metric) for metric in metrics]
        logging.info("Compiling model with optimizer " + str(optimizer) + ", loss " + str(loss) + " and metrics [" +
                     ",".join(metrics_str) + "]")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.model.run_eagerly = run_eagerly
        return self

    def inform(self, dir):
        """
        Creates summary and a visual plot of the model
        :param dir: the directory to store the plot
        :return: the object itself
        """
        logging.info("Creating model plot and summary")
        dir = dir + self._get_model_dir()
        if not os.path.exists(dir):
            os.mkdir(dir)
        try:
            plot_model(self.model, dir + '/network.png', show_shapes=True)
        except Exception as e:
            logging.exception("Can't plot model network.png")
        self.model.summary()
        ModelWeightsLogger().log_model_weights(self.model)
        return self

    def save(self, dir):
        dest_dir = dir + '/' + self._get_model_dir()
        logging.info("Saving model into %s", dest_dir)
        self.model.save(dest_dir)
        logging.info("Saved model")
        return dest_dir

    def prepare_training_data(self, training_dir, output_dir, batch_size, train_percent=0.8, validation_percent=0.1,
                              training_set_limit=None, balance_class_id=None, balance_class_sampling=None):
        training_dataset = self.load_training_set(training_dir=training_dir)
        training_dataset = shuffle(training_dataset)
        test_dataset = None
        if balance_class_sampling is not None:
            training_dataset, test_dataset = \
                self.balance_dataset_with_sampling(training_dataset, balance_class_sampling)
        elif balance_class_id is not None:
            training_dataset, test_dataset = \
                self.balance_dataset_from_class(training_dataset, balance_class_id)
        training_dataset = shuffle(training_dataset)
        if training_set_limit is not None:
            if isinstance(training_dataset, pd.DataFrame):
                training_dataset = training_dataset[:training_set_limit]
            else:
                training_dataset = training_dataset[:training_set_limit]
        dataset_length = len(training_dataset)
        for labels in self.class_ids.values():
            class_len = len(training_dataset[training_dataset['type'].isin(labels)])
            logging.info("%s (%s %%) items of class %s", class_len, class_len / len(training_dataset) * 100, labels)
        train_last_index = int(dataset_length * train_percent)
        train_last_index = train_last_index - train_last_index % batch_size
        validation_last_index = train_last_index + int(dataset_length * validation_percent)
        validation_last_index = validation_last_index if validation_last_index < dataset_length else dataset_length
        test_last_index = dataset_length - 1
        train_dataset_filename = output_dir + "/train_dataset.csv"
        validation_dataset_filename = output_dir + "/validation_dataset.csv"
        test_dataset_filename = output_dir + "/test_dataset.csv"
        logging.info("Storing train and test file names in " + train_dataset_filename + " and " +
                     validation_dataset_filename)
        test_filenames = []
        if os.path.exists(train_dataset_filename):
            os.remove(train_dataset_filename)
        if os.path.exists(validation_dataset_filename):
            os.remove(validation_dataset_filename)
        if os.path.exists(test_dataset_filename):
            os.remove(test_dataset_filename)
        if isinstance(training_dataset, pd.DataFrame):
            train_filenames = training_dataset.iloc[0:train_last_index]
            validation_filenames = training_dataset.iloc[train_last_index:validation_last_index]
            train_filenames.to_csv(train_dataset_filename)
            validation_filenames.to_csv(validation_dataset_filename)
            train_dataset_len = len(train_filenames)
            validation_dataset_len = len(validation_filenames)
            logging.info("Training set is of length %s (%s %%)", len(train_filenames),
                         len(train_filenames) / dataset_length * 100)
            logging.info("Validation set is of length %s (%s %%)", len(validation_filenames),
                         len(validation_filenames) / dataset_length * 100)
            logging.info("Testing set is of length %s (%s %%)", len(test_filenames),
                         len(test_filenames) / dataset_length * 100)
            for labels in self.class_ids.values():
                class_len = len(train_filenames[train_filenames['type'].isin(labels)])
                logging.info("Training set contains %s (%s %%) items of class %s", class_len,
                             class_len / train_dataset_len * 100, labels)
            for labels in self.class_ids.values():
                class_len = len(validation_filenames[validation_filenames['type'].isin(labels)])
                logging.info("Validation set contains %s (%s %%) items of class %s", class_len,
                             class_len / validation_dataset_len * 100, labels)
            if validation_last_index != test_last_index:
                test_filenames = test_dataset if test_dataset is not None else train_filenames.iloc[:0, :].copy()
                test_filenames = pd.concat([test_filenames, training_dataset.iloc[validation_last_index:test_last_index]])
                test_filenames.to_csv(test_dataset_filename)
                test_dataset_len = len(test_filenames)
                for labels in self.class_ids.values():
                    class_len = len(test_filenames[test_filenames['type'].isin(labels)])
                    logging.info("Testing set contains %s (%s %%) items of class %s", class_len,
                                 class_len / test_dataset_len * 100, labels)
            else:
                train_filenames = training_dataset[0:train_last_index]
                validation_filenames = training_dataset[train_last_index:validation_last_index]
        return train_filenames, validation_filenames, test_filenames

    def compute_class_weights(self, train_filenames):
        train_len = len(train_filenames)
        label_counts = {}
        for type, label in self.type_to_label.items():
            class_len = len(train_filenames[train_filenames['type'] == type])
            if label[0] in label_counts:
                label_counts[label[0]] = label_counts[label[0]] + class_len
            else:
                label_counts[label[0]] = class_len
        class_weights = {}
        for label, count in label_counts.items():
            logging.info("Label %s contains %s (%s %%) items", label, count,
                         count / train_len * 100)
            class_weights[label] = 1 / (count / train_len)
        return class_weights

    def train(self, training_dir, output_dir, hyperparams):
        logging.info("Preparing training data with (training_dir," + str(training_dir) +
                     ") (train_percent," + str(hyperparams.train_percent) +
                     ") (test_percent," + str(hyperparams.validation_percent) +
                     ") (training_set_limit," + str(hyperparams.training_set_limit) +
                     ")")
        model_path = output_dir + self._get_model_dir()
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        train_filenames, validation_filenames, test_filenames = \
            self.prepare_training_data(training_dir, model_path, hyperparams.batch_size, hyperparams.train_percent,
                                       hyperparams.validation_percent, hyperparams.training_set_limit,
                                       hyperparams.balance_class_id, hyperparams.balance_class_sampling)
        # The optimizer is executed once for every batch, hence optimizer steps per epoch are
        train_dataset_size = len(train_filenames)
        test_dataset_size = len(validation_filenames)
        steps_per_epoch = int(hyperparams.dataset_iterations_per_epoch * train_dataset_size // hyperparams.batch_size)
        total_steps = steps_per_epoch * hyperparams.epochs
        # if hyperparams.learning_rate_schedule is None:
        #     learning_rate_decay_steps = steps_per_epoch // 2
        #     learning_rate_decay_steps = learning_rate_decay_steps if learning_rate_decay_steps > steps_per_epoch \
        #         else steps_per_epoch
        #     logging.info("Initializing optimizer with learning_rate_decay_steps," + str(learning_rate_decay_steps) +
        #                  ") (gradient_clip_norm," + str(hyperparams.gradient_clip_norm) +
        #                  ")")
        #     powers_for_half_learning_rate_decay = np.log(hyperparams.learning_rate_decay / 2) // \
        #                                           np.log(hyperparams.learning_rate_decay)
        #     hyperparams.learning_rate_schedule = ExponentialRescaleDecay(hyperparams.initial_learning_rate,
        #                                                      decay_steps=learning_rate_decay_steps,
        #                                                      decay_rate=hyperparams.learning_rate_decay,
        #                                                      restore_steps=learning_rate_decay_steps *
        #                                                                    powers_for_half_learning_rate_decay,
        #                                                      restore_rate=1.5,
        #                                                      staircase=True)
        logging.info("Initializing optimizer with (initial_learning_rate," + str(hyperparams.initial_learning_rate) +
                     ")")
        optimizer = tf.keras.optimizers.Adam(hyperparams.learning_rate_schedule
                                             if hyperparams.learning_rate_schedule is not None
                                             else hyperparams.initial_learning_rate,
                                             beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             clipnorm=hyperparams.gradient_clip_norm,
                                             clipvalue=hyperparams.gradient_clip_value)
        # We don't use SparseCategoricalCrossentropy because our targets are one-hot encoded
        loss, accuracy = self.instance_loss_accuracy()
        self.compile(optimizer, hyperparams.custom_loss if hyperparams.custom_loss is not None else loss,
                     metrics=[accuracy] + self.instance_metrics() + hyperparams.metrics,
                     run_eagerly=hyperparams.run_eagerly)
        model_weights_logger = ModelWeightsLogger()
        if not hyperparams.dry_run:
            logging.info("Initializing training and validation generators with (batch_size," + str(hyperparams.batch_size) +
                         ") (self.input_size," + str(self.input_size) +
                         ") (zero_epsilon," + str(hyperparams.zero_epsilon) +
                         ")")
            training_batch_generator = self.instance_generator(train_filenames, training_dir, hyperparams.batch_size,
                                                               self.input_size, self.type_to_label, hyperparams.zero_epsilon)
            validation_batch_generator = self.instance_generator(validation_filenames, training_dir, hyperparams.batch_size,
                                                                 self.input_size, self.type_to_label, hyperparams.zero_epsilon)
            for callback in hyperparams.callbacks:
                if issubclass(callback.__class__, ModelDirDataAwareCallback):
                    callback.set_model_dir(model_path)
                if issubclass(callback.__class__, ValidationDataAwareCallback):
                    callback.set_validation_data(validation_batch_generator)
            hyperparams.callbacks = hyperparams.callbacks + [BatchAwareCsvLogger(model_path + '/training_log.csv', steps_per_epoch),
                                                             GetWeights(model_weights_logger),
                                                             MetricsPlotCallback(self, model_path, steps_per_epoch)]
            if hyperparams.early_stopping_patience > 0 and hyperparams.early_stopping_delta > 0:
                hyperparams.callbacks = hyperparams.callbacks + [ExoMlEarlyStopping(
                    monitor="val_loss",
                    min_delta=hyperparams.early_stopping_delta,
                    patience=hyperparams.early_stopping_patience,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                )]
            model_validation_steps = int(test_dataset_size // hyperparams.batch_size)
            class_weights = hyperparams.class_loss_weights if hyperparams.class_loss_weights is not None \
                else training_batch_generator.class_weights()
            if isinstance(class_weights, str) and 'auto' == class_weights:
                class_weights = self.compute_class_weights(train_filenames)
            logging.info("Initializing training with (epochs," + str(hyperparams.epochs) +
                         ") (steps_per_epoch," + str(steps_per_epoch) +
                         ") (model_validation_steps," + str(model_validation_steps) +
                         ")")
            self.save(output_dir)
            self.__write_hyperparameters(hyperparams, model_path)
            fit_history = self.model.fit(x=training_batch_generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=hyperparams.epochs, verbose=1, class_weight=class_weights,
                           validation_data=validation_batch_generator,
                           validation_steps=model_validation_steps,
                           callbacks=hyperparams.callbacks,
                           use_multiprocessing=hyperparams.cores > 0, workers=1 if hyperparams.cores <= 0
                else hyperparams.cores)
            self.save(output_dir)
        else:
            logging.warning("dry_run was activated and 'training' will not be launched")

    def __write_hyperparameters(self, hyperparameters: HyperParams, output_dir):
        pass
        #only_attributes_hyperparameters = deepcopy(hyperparameters)
        # yaml = ruamel.yaml.YAML()
        # yaml.register_class(HyperParams)
        # # if len(hyperparameters.callbacks) > 0:
        # #     hyperparameters.callbacks = []
        # #     for i, callback in enumerate(hyperparameters.callbacks):
        # #         only_attributes_hyperparameters.callbacks = only_attributes_hyperparameters.callbacks + \
        # #                                                     [str(type(hyperparameters.callbacks[i]))]
        # # if only_attributes_hyperparameters.learning_rate_schedule is not None:
        # #     only_attributes_hyperparameters.learning_rate_schedule = json.dumps(
        # #         dataclasses.asdict(only_attributes_hyperparameters.learning_rate_schedule))
        # # if only_attributes_hyperparameters.custom_loss is not None:
        # #     only_attributes_hyperparameters.custom_loss = str(type(hyperparameters.custom_loss))
        # with open(output_dir + '/hp.yaml', 'w', newline='') as f:
        #     yaml.dump(hyperparameters, f)

    def load_model(self, dir, custom_objects={}):
        custom_objects['ExponentialRescaleDecay'] = ExponentialRescaleDecay
        custom_objects['precision_at_k'] = precision_at_k
        custom_objects['mean_positive_value'] = mean_positive_value
        custom_objects['mean_true_positive_value'] = mean_true_positive_value
        logging.info("Loading model from %s", dir)
        self.set_model(tf.keras.models.load_model(dir,
                                                  custom_objects=custom_objects))
        return self

    def set_model(self, model):
        """
        Stores the model in a class attribute
        :param model: the model to be stored
        """
        self.model = model

    def _get_model_dir(self):
        return self.name + '_model/'
