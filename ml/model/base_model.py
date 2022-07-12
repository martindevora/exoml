import logging
import os
import numpy as np
import tensorflow as tf
import pandas as pd
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


class BaseModel(WithLogging):
    """
    Base class providing standard methods for model building.
    """

    model = None

    def __init__(self, name, input_size, class_ids) -> None:
        super().__init__()
        self.name = name
        self.input_size = input_size
        for class_id in class_ids.keys():
            class_ids[class_id] = class_ids[class_id] if isinstance(class_ids[class_id], (list, np.ndarray)) \
                else [class_ids[class_id]]
        self.class_ids = class_ids

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
    def instance_generator(self, dataset, dir, batch_size, input_sizes, zero_epsilon):
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
                training_set = training_set.append([rows_for_type] * np.floor(sampling).astype(int), ignore_index=True)
                class_expected_size_remainder = np.round(np.mod(sampling, 1) * class_count).astype(int)
                training_set = training_set.append(training_set[training_set['type'].isin(labels)]
                                                   .iloc[0:class_expected_size_remainder], ignore_index=True)
            elif sampling < 1:
                class_expected_size = np.round(class_expected_size).astype(int)
                samples_to_keep = training_set[training_set['type'].isin(labels)].iloc[0:class_expected_size]
                samples_to_remove = training_set[training_set['type'].isin(labels)].iloc[class_expected_size:]
                dropped_samples = dropped_samples.append([samples_to_remove])
                training_set.drop(training_set[training_set['type'].isin(labels)].index, inplace=True)
                training_set = training_set.append([samples_to_keep], ignore_index=True)
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

    def compile(self, optimizer, loss, metrics=None):
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
        plot_model(self.model, dir + '/network.png', show_shapes=True)
        self.model.summary()
        ModelWeightsLogger().log_model_weights(self.model)
        return self

    def save(self, dir):
        dest_dir = dir + '/' + self._get_model_dir()
        logging.info("Saving model into %s", dest_dir)
        self.model.save(dir + self._get_model_dir())
        logging.info("Saved model")

    def prepare_training_data(self, training_dir, output_dir, batch_size, train_percent=0.8, validation_percent=0.1,
                              training_set_limit=None, balance_class_id=None, balance_class_sampling=None):
        training_dataset = self.load_training_set(training_dir=training_dir)
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
            if validation_last_index != test_last_index:
                test_filenames = test_dataset if test_dataset is not None else train_filenames.iloc[:0, :].copy()
                test_filenames = test_filenames.append(training_dataset.iloc[validation_last_index:test_last_index],
                                                       ignore_index=True)
                test_filenames.to_csv(test_dataset_filename)
        else:
            train_filenames = training_dataset[0:train_last_index]
            validation_filenames = training_dataset[train_last_index:validation_last_index]
            with open(train_dataset_filename, 'w') as file:
                file.write('file_name')
                file.write('\n'.join(train_filenames))
            with open(validation_dataset_filename, 'w') as file:
                file.write('file_name')
                file.write('\n'.join(validation_filenames))
            if validation_last_index != test_last_index:
                test_filenames = training_dataset[validation_last_index:test_last_index]
                with open(test_dataset_filename, 'w') as file:
                    file.write('file_name')
                    file.write('\n'.join(test_filenames))
        return train_filenames, validation_filenames, test_filenames

    def train(self, training_dir, output_dir, batch_size=20, epochs=20, initial_learning_rate=0.01,
              dataset_iterations_per_epoch=1, train_percent=0.8,
              validation_percent=0.1, training_set_limit=None, dry_run=True, zero_epsilon=1e-7,
              gradient_clip_norm=0.01, cores=0, metrics=[], callbacks=[], learning_rate_decay=0.98,
              learning_rate_schedule=None, balance_class_id=None, balance_class_sampling=None,
              class_loss_weights=None, early_stopping_delta=0.01, early_stopping_patience=3,
              custom_loss=None):
        logging.info("Preparing training data with (training_dir," + str(training_dir) +
                     ") (train_percent," + str(train_percent) +
                     ") (test_percent," + str(validation_percent) +
                     ") (training_set_limit," + str(training_set_limit) +
                     ")")
        model_path = output_dir + self._get_model_dir()
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        train_filenames, validation_filenames, test_filenames = \
            self.prepare_training_data(training_dir, model_path, batch_size, train_percent, validation_percent,
                                       training_set_limit, balance_class_id, balance_class_sampling)
        # The optimizer is executed once for every batch, hence optimizer steps per epoch are
        train_dataset_size = len(train_filenames)
        test_dataset_size = len(validation_filenames)
        steps_per_epoch = int(dataset_iterations_per_epoch * train_dataset_size // batch_size)
        total_steps = steps_per_epoch * epochs
        if learning_rate_schedule is None:
            learning_rate_decay_steps = steps_per_epoch // 2
            learning_rate_decay_steps = learning_rate_decay_steps if learning_rate_decay_steps > steps_per_epoch else steps_per_epoch
            logging.info("Initializing optimizer with learning_rate_decay_steps," + str(learning_rate_decay_steps) +
                         ") (gradient_clip_norm," + str(gradient_clip_norm) +
                         ")")
            powers_for_half_learning_rate_decay = np.log(learning_rate_decay / 2) // np.log(learning_rate_decay)
            learning_rate_schedule = ExponentialRescaleDecay(initial_learning_rate,
                                                             decay_steps=learning_rate_decay_steps,
                                                             decay_rate=learning_rate_decay,
                                                             restore_steps=learning_rate_decay_steps * powers_for_half_learning_rate_decay,
                                                             restore_rate=1.5,
                                                             staircase=True)
        logging.info("Initializing optimizer with (initial_learning_rate," + str(initial_learning_rate) + ")")
        optimizer = tf.keras.optimizers.Adam(learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             clipnorm=gradient_clip_norm)
        # We don't use SparseCategoricalCrossentropy because our targets are one-hot encoded
        loss, accuracy = self.instance_loss_accuracy()
        self.compile(optimizer, custom_loss if custom_loss is not None else loss,
                     metrics=[accuracy] + self.instance_metrics() + metrics)
        model_weights_logger = ModelWeightsLogger()
        if not dry_run:
            logging.info("Initializing training and validation generators with (batch_size," + str(batch_size) +
                         ") (self.input_size," + str(self.input_size) +
                         ") (zero_epsilon," + str(zero_epsilon) +
                         ")")
            lcs_dir = training_dir + '/lcs/'
            training_batch_generator = self.instance_generator(train_filenames, lcs_dir, batch_size,
                                                               self.input_size, zero_epsilon)
            validation_batch_generator = self.instance_generator(validation_filenames, lcs_dir, batch_size,
                                                                 self.input_size, zero_epsilon)
            csv_log = model_path + '/training_log.csv'
            logging.info("Initializing csv logger with (csv_log," + str(csv_log) +
                         ") (steps_per_epoch," + str(steps_per_epoch) +
                         ")")
            csv_logger = BatchAwareCsvLogger(csv_log, steps_per_epoch)
            model_path = output_dir + self._get_model_dir()
            logging.info("Initializing model checkpoint with (model_path," + str(model_path) +
                         ") (steps_per_epoch," + str(steps_per_epoch) +
                         ")")
            for callback in callbacks:
                if issubclass(callback.__class__, ModelDirDataAwareCallback):
                    callback.set_model_dir(model_path)
                if issubclass(callback.__class__, ValidationDataAwareCallback):
                    callback.set_validation_data(validation_batch_generator)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                verbose=1,
                save_weights_only=False,
                save_freq='epoch')
            early_stopping_callback = ExoMlEarlyStopping(
                monitor="val_loss",
                min_delta=early_stopping_delta,
                patience=early_stopping_patience,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
            )
            get_weights_callback = GetWeights(model_weights_logger)
            model_validation_steps = int(test_dataset_size // batch_size)
            class_weights = class_loss_weights if class_loss_weights is not None \
                else training_batch_generator.class_weights()
            logging.info("Initializing training with (epochs," + str(epochs) +
                         ") (steps_per_epoch," + str(steps_per_epoch) +
                         ") (model_validation_steps," + str(model_validation_steps) +
                         ")")
            self.save(output_dir)
            fit_history = self.model.fit(x=training_batch_generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=epochs, verbose=1, class_weight=class_weights,
                           validation_data=validation_batch_generator,
                           validation_steps=model_validation_steps,
                           callbacks=[csv_logger, cp_callback, get_weights_callback, early_stopping_callback,
                                      MetricsPlotCallback(self, model_path, steps_per_epoch)] +
                                      callbacks,
                           use_multiprocessing=cores > 0, workers=1 if cores <= 0 else cores)
            self.save(output_dir)
        else:
            logging.warning("dry_run was activated and 'training' will not be launched")

    def load_model(self, dir):
        self.set_model(tf.keras.models.load_model(dir,
                                                  custom_objects={'ExponentialRescaleDecay': ExponentialRescaleDecay}))
        return self

    def set_model(self, model):
        """
        Stores the model in a class attribute
        :param model: the model to be stored
        """
        self.model = model

    def _get_model_dir(self):
        return self.name + '_model/'


class CategoricalPredictionSetStats:
    """
    Statistics of categorical predictions.
    """
    def __init__(self, tp, fp, fn, accuracy, precision, recall) -> None:
        super().__init__()
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
