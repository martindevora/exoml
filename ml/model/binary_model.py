import logging

from keras.metrics import Precision, Recall, AUC, BinaryAccuracy, RecallAtPrecision
from tensorflow_addons.metrics import F1Score

from ml.loss.cross_entropy import binary_focal_loss
from ml.model.base_model import BaseModel
import pandas as pd
import matplotlib.pyplot as plt


class BinaryModel(BaseModel):
    def __init__(self, name, input_size, class_ids) -> None:
        super().__init__(name, input_size, class_ids)

    def instance_loss_accuracy(self):
        return binary_focal_loss, BinaryAccuracy()

    def instance_metrics(self):
        return [Precision(), Recall(), RecallAtPrecision(precision=0.99), F1Score(num_classes=1), AUC(name="roc_auc"),
                AUC(curve="PR", name="pr_auc")]

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        logging.info("Plotting metrics")
        training_log_df = pd.read_csv(model_dir + '/training_log.csv')
        training_log_df['batch_total'] = training_log_df['batch'] + (training_log_df['epoch'] * steps_per_epoch)
        validation_log_df = training_log_df.dropna()
        fig, axs = plt.subplots(2, 3, figsize=(25, 7), constrained_layout=True)
        fig.suptitle(self.name + ' Model Metrics', fontsize=14)
        axs[0][0].plot(training_log_df['batch_total'], training_log_df['learning_rate'])
        axs[0][1].plot(training_log_df['batch_total'], training_log_df['loss'], color='blue', label='Train loss')
        if with_validation:
            axs[0][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_loss'], color='red',
                           label='Val. loss')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['precision'], color='orange', label='Train Prec.')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['recall'], color='cyan', label='Train Recall.')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['r@p'], color='mediumaquamarine', label='Train r@p.')
        axs[1][1].plot(training_log_df['batch_total'], training_log_df['accuracy'], color='black', label='Train Acc.')
        axs[1][1].plot(training_log_df['batch_total'], training_log_df['f1_score'], color='orange', label='Train F1')
        axs[1][1].plot(training_log_df['batch_total'], training_log_df['roc_auc'], color='cyan', label='Train ROC AUC')
        axs[1][1].plot(training_log_df['batch_total'], training_log_df['pr_auc'], color='mediumaquamarine',
                       label='Train PR AUC')
        if with_validation:
            axs[1][0].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_precision'], color='red',
                           label='Val. prec.')
            axs[1][0].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_recall'], color='blue',
                           label='Val. recall.')
            axs[1][0].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_r@p'], color='green',
                           label='Val. r@p')
            axs[1][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_accuracy'], color='gray',
                           label='Val. acc.')
            axs[1][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_f1_score'], color='red',
                           label='Val. F1')
            axs[1][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_roc_auc'], color='blue',
                           label='Val. ROC AUC')
            axs[1][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_pr_auc'], color='green',
                           label='Val. PR AUC')
        axs[0][0].set_xlabel("Batch")
        secx = axs[0][0].secondary_xaxis('top',
                                         functions=(lambda x: x / steps_per_epoch, lambda x: x / steps_per_epoch))
        secx.set_xlabel('Epoch')
        axs[0][0].set_ylabel("Learning rate")
        axs[0][0].legend(loc='upper right')
        axs[0][1].set_xlabel("Batch")
        secx = axs[0][1].secondary_xaxis('top',
                                         functions=(lambda x: x / steps_per_epoch, lambda x: x / steps_per_epoch))
        secx.set_xlabel('Epoch')
        axs[0][1].set_ylabel("Loss")
        axs[0][1].legend(loc='upper right')
        axs[1][0].set_xlabel("Batch")
        secx = axs[1][0].secondary_xaxis('top',
                                         functions=(lambda x: x / steps_per_epoch, lambda x: x / steps_per_epoch))
        secx.set_xlabel('Epoch')
        axs[1][0].set_ylabel("Precision/Recall/R@P")
        axs[1][0].legend(loc='upper left')
        axs[1][1].set_xlabel("Batch")
        secx = axs[1][1].secondary_xaxis('top',
                                         functions=(lambda x: x / steps_per_epoch, lambda x: x / steps_per_epoch))
        secx.set_xlabel('Epoch')
        axs[1][1].set_ylabel("Accuracy/F1/AUCs")
        axs[1][1].legend(loc='upper left')
        plt.savefig(model_dir + '/metrics.png')
        plt.close(fig)
        plt.clf()
        logging.info("Plotted metrics")
