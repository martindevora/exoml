import logging
import os
from threading import Thread
from typing import List

import numpy
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
import keras
import numpy as np
import tensorflow as tf
import time

from keras.src.optimizers import Adam
from keras.utils import plot_model
from keras.src.layers import LeakyReLU
from numpy import ndarray
from sklearn.utils import shuffle

from exoml.ml.callback.learning_rate import SWA, MultiOptimizer
from exoml.ml.model.base_model import HyperParams

def weighted_binary_crossentropy_param(weight=1):
    def weighted_binary_crossentropy(y_true, y_pred, weight=weight):
        y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(logloss, axis=-1)
    return weighted_binary_crossentropy


class SANTO:
    """
    Self-Attention Neural Network for Transiting Objects
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = "SANTO"
        self.steps_per_epoch = 1

    def load(self, dir, hyperparams: HyperParams):
        self.model = tf.keras.models.load_model(dir,
                           custom_objects={'loss': self.loss(hyperparams)})

    def loss(self, hyperparams: HyperParams):
        return weighted_binary_crossentropy_param(hyperparams.class_loss_weights[0])

    def inform(self, dir):
        """
        Creates summary and a visual plot of the model
        :param dir: the directory to store the plot
        :return: the object itself
        """
        print("Creating model plot and summary")
        if not os.path.exists(dir):
            os.mkdir(dir)
        try:
            plot_model(self.model, dir + '/network.png', show_shapes=True)
        except Exception as e:
            print("Can't plot model network.png")
        self.model.summary()
        return self

    def train(self, lcs_dir, model_dir, hyperparams: HyperParams, validation_set_percent=0.1, input_dim=500,
              cores=os.cpu_count() // 2, load_checkpoint=False):
        print("Begin training")
        tf.keras.backend.clear_session()
        input_layer = keras.Input(shape=(input_dim, 1), name="input")
        # outputs = TransformerClassifier(transformer_input_size=input_dim, patch_size=input_dim,
        #                                     num_heads=1, mlp_dim=input_dim, hyperparams=hyperparams, num_blocks=1,
        #                                     classes=1)(input_layer)
        leaky_relu_alpha = 0.01
        flux_branch: keras.layers.Layer = input_layer
        # flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=16, kernel_size=90, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=10, strides=1)(flux_branch)
        # flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=hyperparams.spatial_dropout_rate)(flux_branch)
        conv1_branch: keras.layers.Layer = keras.layers.Flatten()(flux_branch)
        conv1_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha),
                                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(conv1_branch)
        conv1_branch = keras.layers.Dropout(hyperparams.dropout_rate)(conv1_branch)
        flux_branch = keras.layers.Conv1D(filters=32, kernel_size=30, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=4, strides=1)(flux_branch)
        # flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=hyperparams.spatial_dropout_rate)(flux_branch)
        conv2_branch: keras.layers.Layer = keras.layers.Flatten()(flux_branch)
        conv2_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(conv2_branch)
        conv2_branch = keras.layers.Dropout(hyperparams.dropout_rate)(conv2_branch)
        flux_branch = keras.layers.Conv1D(filters=64, kernel_size=15, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=2)(flux_branch)
        # flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.SpatialDropout1D(rate=hyperparams.spatial_dropout_rate)(flux_branch)
        conv3_branch: keras.layers.Layer = keras.layers.Flatten()(flux_branch)
        conv3_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(conv3_branch)
        conv3_branch = keras.layers.Dropout(hyperparams.dropout_rate)(conv3_branch)
        flux_branch = keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=2, strides=2)(flux_branch)
        conv4_branch: keras.layers.Layer = keras.layers.Flatten()(flux_branch)
        conv4_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(conv4_branch)
        conv4_branch = keras.layers.Dropout(hyperparams.dropout_rate)(conv4_branch)
        flux_branch = keras.layers.Concatenate(axis=1)([
            conv1_branch, conv2_branch, conv3_branch, conv4_branch])
        flux_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha),
                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(flux_branch)
        flux_branch = keras.layers.Dropout(hyperparams.dropout_rate)(flux_branch)
        flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha),
                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=hyperparams.l1_regularization, l2=hyperparams.l2_regularization))(flux_branch)
        flux_branch = keras.layers.Dropout(hyperparams.dropout_rate)(flux_branch)
        outputs: keras.layers.Layer = keras.layers.Dense(1, activation="sigmoid")(flux_branch)
        print("Preparing data sets")
        target_files: List[str] = os.listdir(lcs_dir)
        target_files = shuffle(target_files)
        target_files_partition_index = int(len(target_files) * (1 - validation_set_percent))
        train_target_files = target_files[0:target_files_partition_index]
        validation_target_files = target_files[target_files_partition_index + 1:]
        train_dataset_size = len(train_target_files)
        steps_per_epoch = int(hyperparams.dataset_iterations_per_epoch * train_dataset_size // hyperparams.batch_size)
        optimizer: keras.optimizers.Optimizer = self.build_optimizer(hyperparams, steps_per_epoch=steps_per_epoch)
        print("Compiling model")
        self.model: keras.Model = keras.Model(inputs=input_layer, outputs=outputs)
        self.model.compile(optimizer=optimizer,
                                  loss=self.loss(hyperparams),
                                  metrics=['accuracy', 'precision', 'recall'])
        self.model.run_eagerly = hyperparams.run_eagerly
        print("Model summary")
        self.inform(model_dir)
        print("Preparing checkpoints")
        checkpoint_path = f"{model_dir}_chk"
        ckpt: tf.train.Checkpoint = tf.train.Checkpoint(transformer=self.model, optimizer=optimizer)
        ckpt_manager: tf.train.CheckpointManager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=hyperparams.epochs)
#        if load_checkpoint and ckpt_manager.latest_checkpoint:
#            print("Restoring last checkpoint.")
#            ckpt.restore(ckpt_manager.latest_checkpoint)
#            print("Last checkpoint restored.")
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        pr99s = []
        val_losses = []
        val_accuracies = []
        val_precisions = []
        val_recalls = []
        val_pr99s = []
        file_flux_len = len(np.loadtxt(f'{lcs_dir}/{train_target_files[0]}', delimiter=',')[1])
        steps_per_file = file_flux_len - hyperparams.batch_size
        steps_per_file = steps_per_file if steps_per_file >= 1 else 1
        self.steps_per_epoch = train_dataset_size * steps_per_file
        training_log_df = pd.DataFrame(columns=['epoch', 'batch', 'loss', 'lr', 'accuracy', 'p@r99', 'precision', 'recall',
                                                'val_accuracy', 'val_p@r99', 'val_precision', 'val_recall'])
        for epoch in range(hyperparams.epochs):
            print(f"Epoch {epoch + 1}/{hyperparams.epochs}")
            train_target_files = shuffle(train_target_files)
            start = time.time()
            epoch_losses, epoch_accuracies, epoch_precisions, epoch_recalls, epoch_pr99s = \
                self.run_epoch(epoch, lcs_dir, model_dir, train_target_files, hyperparams, input_dim, self.model,
                               optimizer, training=True, cores=cores, training_log_df=training_log_df)
            losses = losses + epoch_losses
            accuracies = accuracies + epoch_accuracies
            precisions = precisions + epoch_precisions
            recalls = recalls + epoch_recalls
            pr99s = pr99s + epoch_pr99s
            val_epoch_losses, val_epoch_accuracies, val_epoch_precisions, val_epoch_recalls, val_epoch_pr99s = \
                self.run_epoch(epoch, lcs_dir, model_dir, validation_target_files, hyperparams, input_dim, self.model,
                               optimizer, cores=cores, training=False)
            val_losses = val_losses + val_epoch_losses
            val_accuracies = val_accuracies + val_epoch_accuracies
            val_precisions = val_precisions + val_epoch_precisions
            val_recalls = val_recalls + val_epoch_recalls
            val_pr99s = val_pr99s + val_epoch_pr99s
            # Checkpoint the model on every epoch
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} in {}".format(epoch + 1, ckpt_save_path))
            print("Time for 1 epoch: {} secs\n".format(time.time() - start))
            self.plot_metrics(model_dir, self.steps_per_epoch, False)
        print(f"Saving model into {model_dir}")
        self.model.save(model_dir)
        print("Saved model")
        return losses, accuracies

    def compute_initial_lr(self, hyperparams: HyperParams):
        if hyperparams.lr_progression != 1:
            initial_lr = hyperparams.initial_learning_rate
        else:
            initial_lr = hyperparams.learning_rate_schedule \
                if hyperparams.learning_rate_schedule is not None else hyperparams.initial_learning_rate
        return initial_lr

    def build_optimizer(self, hyperparams, steps_per_epoch):
        initial_lr = self.compute_initial_lr(hyperparams)
        optimizer = self.build_swa_optimizer(hyperparams, initial_lr, steps_per_epoch)
        if hyperparams.lr_progression != 1:
            optimizer.progressive_lr_factor = 1
            optimizers_and_layers = []
            standard_layers = []
            progressive_lr_factor = 1
            for layer in self.model.layers:
                if 'final' in layer.name:
                    progressive_lr_factor = progressive_lr_factor * hyperparams.lr_progression
                    progressive_optimizer = self.build_swa_optimizer(hyperparams, progressive_lr_factor * initial_lr, steps_per_epoch)
                    progressive_optimizer.progressive_lr_factor = progressive_lr_factor
                    optimizers_and_layers = optimizers_and_layers + [
                        (progressive_optimizer, layer)]
                else:
                    standard_layers = standard_layers + [layer]
            optimizers_and_layers = [(optimizer, standard_layers)] + optimizers_and_layers
            optimizer = MultiOptimizer(optimizers_and_layers)
        return optimizer

    def build_swa_optimizer(self, hyperparams: HyperParams, lr, steps_per_epoch):
        if hyperparams.stochastic_weight_average_wait > 0:
            optimizer = tf.keras.optimizers.legacy.Adam(lr,
                                                        beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                                        clipnorm=hyperparams.gradient_clip_norm,
                                                        clipvalue=hyperparams.gradient_clip_value)
            optimizer = SWA(optimizer,
                                           start_averaging=steps_per_epoch * hyperparams.stochastic_weight_average_wait,
                                           average_period=steps_per_epoch)
            if isinstance(lr, (int, float)):
                optimizer.lr = lr
            else:
                optimizer.lr = hyperparams.initial_learning_rate
        else:
            optimizer = Adam(lr,
                                                 beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                                 clipnorm=hyperparams.gradient_clip_norm,
                                                 clipvalue=hyperparams.gradient_clip_value)
        return optimizer

    def run_epoch(self, epoch, lcs_dir, model_dir, target_files, hyperparams, input_dim, transformer_model, optimizer,
                  training=False, cores=os.cpu_count() // 2, training_log_df=None):
        training_preparation_threads: list[Thread] = []
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        pr99s = []
        accuracy: keras.metrics.Metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        precision: keras.metrics.Metric = tf.keras.metrics.Precision(name="precision")
        recall: keras.metrics.Metric = tf.keras.metrics.Recall(name="recall")
        pr99: keras.metrics.Metric = tf.keras.metrics.PrecisionAtRecall(recall=0.99, name="p@r99")
        step = 0
        for file_index, last_target_file in enumerate(target_files):
            last_target_file_pos: int = 0
            flux = np.loadtxt(f'{lcs_dir}/{last_target_file}', delimiter=',')
            if np.nanmin(flux[1]) == 1:
                continue
            train_batch: ndarray = np.zeros((hyperparams.batch_size, input_dim))
            train_tags: ndarray = np.zeros((hyperparams.batch_size, 1))
            iteration: int = 0
            data_remaining: bool = True
            while data_remaining:
                # training_preparation_threads: list[Thread] = []
                # for core in range(0, cores):
                #     training_preparation_threads =
                batch_index = iteration // hyperparams.batch_size
                iteration_index = iteration % hyperparams.batch_size
                train_batch[iteration_index] = flux[0][last_target_file_pos:last_target_file_pos + input_dim] / 2
                train_tags[iteration_index] = flux[1][last_target_file_pos + input_dim // 2 - 1]
                oot_mask = np.abs(train_tags[iteration_index] - 1) < 1e-6
                train_tags[iteration_index][oot_mask] = 0
                train_tags[iteration_index][~oot_mask] = 1
                last_target_file_pos = last_target_file_pos + 1
                data_remaining = last_target_file_pos < flux.shape[1] - input_dim
                if (iteration + 1) % hyperparams.batch_size == 0 or not data_remaining:
                    train_tags_tensor = tf.convert_to_tensor(train_tags, dtype=np.float32)
                    train_batch_tensor = tf.convert_to_tensor(train_batch, dtype=np.float32)
                    train_tags_tensor = tf.reshape(train_tags_tensor, [hyperparams.batch_size, 1])
                    if training:
                        with tf.GradientTape() as tape:
                            tape.watch(transformer_model.trainable_variables)
                            flux_predictions = transformer_model(train_batch)
                            loss = transformer_model.compute_loss(train_batch_tensor, train_tags_tensor, flux_predictions)
                        losses.append(loss.numpy())
                        gradients = tape.gradient(loss, transformer_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, transformer_model.trainable_variables))
                        for gradient in gradients:
                            if np.any(gradient.numpy() == 0):
                                print("There are gradients with zero values")
                                break
                        if np.all(flux_predictions == 0):
                            print("All predictions are 0")
                        if np.all(flux_predictions == 1):
                            print("All predictions are 1")
                        step = step + 1
                    else:
                        flux_predictions = transformer_model.predict_on_batch(train_batch)
                    # flux_predictions_reshape = tf.reshape(flux_predictions, [hyperparams.batch_size, input_dim])
                    train_tags_tensor = tf.reshape(train_tags_tensor, [hyperparams.batch_size, 1])
                    accuracy(train_tags_tensor, flux_predictions)
                    precision(train_tags_tensor, flux_predictions)
                    recall(train_tags_tensor, flux_predictions)
                    pr99(train_tags_tensor, flux_predictions)
                    accuracies.append(accuracy.result().numpy())
                    precisions.append(precision.result().numpy())
                    recalls.append(recall.result().numpy())
                    pr99s.append(pr99.result().numpy())
                    if training:
                        print("Epoch {} File {} Batch {} Loss {} Accuracy {} Precision {} Recall {} P@R99 {}"
                              .format(epoch + 1, file_index, step, loss, accuracy.result(),
                                      precision.result(), recall.result(), pr99.result()))
                        if training_log_df is not None:
                            if isinstance(self.model.optimizer, MultiOptimizer):
                                learning_rate = K.eval(K.eval(self.model.optimizer.optimizer_specs[0]['optimizer'].lr))
                            else:
                                learning_rate = K.eval(self.model.optimizer._decayed_lr(tf.float32))
                            training_log_df = pd.concat([training_log_df, pd.DataFrame.from_dict(
                                {'lr': [learning_rate], 'epoch': [epoch], 'batch': [step], 'loss': [loss.numpy()],
                                 'accuracy': [accuracy.result().numpy()], 'precision': [precision.result().numpy()],
                                 'recall': [recall.result().numpy()],
                                 'p@r99': [pr99.result().numpy()]})], ignore_index=True)
                            training_log_df.to_csv(model_dir + '/training_log.csv', index=False)
                            #self.plot_metrics(model_dir, self.steps_per_epoch, False)
                        accuracy.reset_state()
                        precision.reset_state()
                        recall.reset_state()
                        pr99.reset_state()
                iteration = iteration + 1
        if not training: 
            print(f"Validation Epoch {epoch + 1} Accuracy {accuracy.result().numpy()} "
                  f"Precision {precision.result().numpy()} Recall {recall.result().numpy()} P@R99 "
                  f"{pr99.result().numpy()}")
            if training_log_df is not None:
                if isinstance(self.model.optimizer, MultiOptimizer):
                    learning_rate = K.eval(K.eval(self.model.optimizer.optimizer_specs[0]['optimizer'].lr))
                else:
                                learning_rate = K.eval(self.model.optimizer._decayed_lr(tf.float32))
                training_log_df = pd.concat([training_log_df, pd.DataFrame.from_dict(
                    {'lr': [learning_rate], 'epoch': [epoch], 'batch': [0],
                     'val_accuracy': [accuracy.result().numpy()], 'val_precision': [precision.result().numpy()],
                     'val_recall': [recall.result().numpy()], 'val_p@r99': [pr99.result().numpy()]})], ignore_index=True)
                training_log_df.to_csv(model_dir + '/training_log.csv', index=False)
        return losses, accuracies, precisions, recalls, pr99s

    def plot_metrics(self, model_dir, steps_per_epoch, with_validation=False):
        logging.info("Plotting metrics")
        training_log_df = pd.read_csv(model_dir + '/training_log.csv')
        training_log_df['batch_total'] = training_log_df['batch'] + (training_log_df['epoch'] * steps_per_epoch)
        validation_log_df = training_log_df.dropna(subset=['val_precision', 'val_recall', 'val_p@r99', 'val_accuracy'])
        fig, axs = plt.subplots(2, 2, figsize=(25, 7), constrained_layout=True)
        fig.suptitle(self.name + ' Model Metrics', fontsize=14)
        axs[0][0].plot(training_log_df['batch_total'], training_log_df['lr'])
        axs[0][1].plot(training_log_df['batch_total'], training_log_df['loss'], color='blue', label='Train loss')
        if with_validation:
            axs[0][1].plot(validation_log_df['epoch'] * steps_per_epoch, validation_log_df['val_loss'], color='red',
                           label='Val. loss')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['precision'], color='red', label='Train Prec.')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['recall'], color='blue', label='Train Recall.')
        axs[1][0].plot(training_log_df['batch_total'], training_log_df['p@r99'], color='green', label='Train p@r.')
        axs[1][1].plot(training_log_df['batch_total'], training_log_df['accuracy'], color='black', label='Train Acc.')
        if with_validation:
            axs[1][0].plot(validation_log_df['batch_total'], validation_log_df['val_precision'],
                           color='red', label='Val. prec.', linestyle='--')
            axs[1][0].plot(validation_log_df['batch_total'], validation_log_df['val_recall'],
                           color='blue', label='Val. recall.', linestyle='--')
            axs[1][0].plot(validation_log_df['batch_total'], validation_log_df['val_p@r99'],
                           color='green', label='Val. p@r', linestyle='--')
            axs[1][1].plot(validation_log_df['batch_total'], validation_log_df['val_accuracy'],
                           color='black', label='Val. acc.', linestyle='--')
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
        axs[1][0].set_ylabel("Precision/Recall/P@R")
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

# SANTO().train("/data/scratch/ml/santo/training_data/", "/data/scratch/exoml/SANTO",
#               hyperparams=HyperParams(batch_size=100, epochs=10, run_eagerly=True))
