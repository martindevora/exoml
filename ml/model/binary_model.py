from keras.metrics import Precision, Recall, AUC, BinaryAccuracy, RecallAtPrecision
from tensorflow_addons.metrics import F1Score

from ml.loss.cross_entropy import binary_focal_loss
from ml.model.base_model import BaseModel
import pandas as pd
import matplotlib.pyplot as plt


class BinaryModel(BaseModel):
    def __init__(self, name, input_size) -> None:
        super().__init__(name, input_size)

    def instance_loss_accuracy(self):
        return binary_focal_loss, BinaryAccuracy()

    def instance_metrics(self):
        return [Precision(), Recall(), RecallAtPrecision(precision=0.99), F1Score(), AUC(name="roc_auc"),
                AUC(curve="PR", name="pr_auc")]

    def plot_metrics(self, model_dir, steps_per_epoch):
        training_log_df = pd.read_csv(model_dir + '/training_log.csv')
        validation_log_df = pd.read_csv(model_dir + '/val_log.csv')
        fig, axs = plt.figure((3, 3), (15, 15))
        axs[0][0].plot(training_log_df['batch'], training_log_df['learning_rate'])
        axs[0][0].set_xlabel("Batch no.")
        axs[0][0].set_ylabel("Learning rate")
        axs[0][0].legend(loc='upper right')
        axs[0][1].plot(training_log_df['batch'], training_log_df['loss'], color='blue', label='Train loss')
        axs[0][1].plot(validation_log_df['epoch'] * steps_per_epoch, training_log_df['loss'], color='red',
                       label='Val. loss')
        axs[0][1].set_xlabel("Batch no.")
        axs[0][1].set_ylabel("Loss")
        axs[0][1].legend(loc='upper right')
        axs[0][2].plot(training_log_df['batch'], training_log_df['accuracy'], color='blue', label='Train Acc.')
        axs[0][2].plot(validation_log_df['epoch'] * steps_per_epoch, training_log_df['accuracy'], color='red',
                       label='Val. acc.')
        axs[0][2].set_xlabel("Batch no.")
        axs[0][2].set_ylabel("Accuracy")
        axs[0][2].legend(loc='upper left')