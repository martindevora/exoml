import logging
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

    def set_model(self, model):
        """
        Stores the model in a class attribute
        :param model: the model to be stored
        """
        self.model = model
