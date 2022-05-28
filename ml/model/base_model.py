import logging
import os

import tensorflow as tf
import pandas as pd
from abc import abstractmethod

from keras.utils.vis_utils import plot_model

from ml.log.get_weights_logger import ModelWeightsLogger
from ml.log.with_logging import WithLogging


class BaseModel(WithLogging):
    """
    Base class providing standard methods for model building.
    """

    model = None

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def build(self, **kwargs):
        """
        Should be used to create all the model layers and related capabilities to be used in the training and subsequent
        tasks.
        :param kwargs:
        """
        pass

    @abstractmethod
    def train(self, **kwargs):
        """
        Launches the model training with the case-specific kwargs
        :param kwargs: the case-specific arguments
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

    def inform(self, image_file_name):
        """
        Creates summary and a visual plot of the model
        :param image_file_name: the file name of the visual plot
        :return: the object itself
        """
        logging.info("Creating model plot and summary")
        plot_model(self.model, image_file_name, show_shapes=True)
        self.model.summary()
        ModelWeightsLogger().log_model_weights(self.model)
        return self

    def prepare_training_data(self, training_dir, output_dir, train_percent=0.8, validation_percent=0.1,
                              training_set_limit=None):
        training_dataset = self.load_training_set(training_dir=training_dir)
        if training_set_limit is not None:
            if isinstance(training_dataset, pd.DataFrame):
                training_dataset = training_dataset[:training_set_limit]
            else:
                training_dataset = training_dataset[:training_set_limit]
        dataset_length = len(training_dataset)
        train_last_index = int(dataset_length * train_percent)
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
                test_filenames = training_dataset.iloc[validation_last_index:test_last_index]
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

    def set_model(self, model):
        """
        Stores the model in a class attribute
        :param model: the model to be stored
        """
        self.model = model

    def load_model(self, dir):
        self.set_model(tf.keras.models.load_model(dir))
        return self


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
