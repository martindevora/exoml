import logging
import os
import pathlib

import keras
import tensorflow as tf
from keras import Model
from keras.layers import LeakyReLU
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from keras.utils import losses_utils
from sklearn.utils import shuffle
from tensorflow.python.keras.utils.vis_utils import plot_model

from ml import thread_config
from ml.callback.batch_aware_csvlogger import BatchAwareCsvLogger
from ml.callback.get_weights import GetWeights
from ml.generator.detrend_generator import DetrendModelGenerator
from ml.log.get_weights_logger import ModelWeightsLogger
from ml.log.with_logging import WithLogging
from ml.metrics.fixed_mean import FixedMean


thread_config.setup_threads(5)


class AutoEncoder(WithLogging):
    def __init__(self, input_size=(20610, 8)):
        super().__init__()
        self.input_size = input_size

    def build(self):
        # (time, flux, flux_err, centroidx, centroidy, motionx, motiony, bck)
        autoencoder_layer1_strides = 10
        autoencoder_layer1_filters = 5000
        autoencoder_layer1_ks = 100
        autoencoder_layer2_strides = 5
        autoencoder_layer2_filters = 1250
        autoencoder_layer2_ks = 33
        autoencoder_layer3_strides = 5
        autoencoder_layer3_filters = 420
        autoencoder_layer3_ks = 15
        autoencoder_layer4_strides = 2
        autoencoder_layer4_filters = 128
        autoencoder_layer4_ks = 9
        autoencoder_layer5_strides = 2
        autoencoder_layer5_filters = 64
        autoencoder_layer5_ks = 7
        autoencoder_layer6_strides = 1
        autoencoder_layer6_filters = 32
        autoencoder_layer6_ks = 5
        input = keras.Input(shape=(self.input_size))
        self.batch_norm_input = keras.layers.normalization.batch_normalization.BatchNormalization()(input)
        self.enc_layer1 = keras.layers.SpatialDropout1D(rate=0.1)(input)
        self.enc_layer1_r = keras.layers.Conv1D(filters=autoencoder_layer1_filters, kernel_size=autoencoder_layer1_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer1)
        self.enc_layer1 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer1_r)
        self.enc_layer1 = keras.layers.MaxPooling1D(pool_size=50, strides=autoencoder_layer1_strides, padding="same")(self.enc_layer1)
        self.enc_layer1 = keras.layers.Dropout(rate=0.1)(self.enc_layer1)
        self.enc_layer2_r = keras.layers.Conv1D(filters=autoencoder_layer2_filters, kernel_size=autoencoder_layer2_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer1)
        self.enc_layer2 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer2_r)
        self.enc_layer2 = keras.layers.MaxPooling1D(pool_size=20, strides=autoencoder_layer2_strides, padding="same")(self.enc_layer2)
        self.enc_layer2 = keras.layers.Dropout(rate=0.1)(self.enc_layer2)
        self.enc_layer3_r = keras.layers.Conv1D(filters=autoencoder_layer3_filters, kernel_size=autoencoder_layer3_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer2)
        self.enc_layer3 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer3_r)
        self.enc_layer3 = keras.layers.MaxPooling1D(pool_size=15, strides=autoencoder_layer3_strides, padding="same")(self.enc_layer3)
        self.enc_layer3 = keras.layers.Dropout(rate=0.1)(self.enc_layer3)
        self.enc_layer4_r = keras.layers.Conv1D(filters=autoencoder_layer4_filters, kernel_size=autoencoder_layer4_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer3)
        self.enc_layer4 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer4_r)
        self.enc_layer4 = keras.layers.MaxPooling1D(pool_size=10, strides=autoencoder_layer4_strides, padding="same")(self.enc_layer4)
        self.enc_layer4 = keras.layers.Dropout(rate=0.1)(self.enc_layer4)
        self.enc_layer5_r = keras.layers.Conv1D(filters=autoencoder_layer5_filters, kernel_size=autoencoder_layer5_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer4)
        self.enc_layer5 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer5_r)
        self.enc_layer5 = keras.layers.MaxPooling1D(pool_size=4, strides=autoencoder_layer5_strides, padding="same")(self.enc_layer5)
        self.enc_layer5 = keras.layers.Dropout(rate=0.1)(self.enc_layer5)
        self.enc_layer6_r = keras.layers.Conv1D(filters=autoencoder_layer6_filters, kernel_size=autoencoder_layer6_ks,
                                                padding="same", activation=LeakyReLU(0.01))(self.enc_layer5)
        self.enc_layer6 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.enc_layer6_r)
        self.enc_layer6 = keras.layers.MaxPooling1D(pool_size=3, strides=autoencoder_layer6_strides, padding="same")(self.enc_layer6)
        self.enc_layer6 = keras.layers.Dropout(rate=0.1)(self.enc_layer6)
        self.dec_layer6 = keras.layers.UpSampling1D(autoencoder_layer6_strides)(self.enc_layer6)
        self.dec_layer6 = keras.layers.Conv1DTranspose(filters=autoencoder_layer6_filters, kernel_size=autoencoder_layer6_ks,
                                                       padding="same", activation=LeakyReLU(0.01))(self.dec_layer6)
        self.dec_layer6 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer6)
        self.dec_layer6 = keras.layers.Add()([self.enc_layer6_r, self.dec_layer6])
        self.dec_layer5 = keras.layers.UpSampling1D(autoencoder_layer5_strides)(self.dec_layer6)
        self.dec_layer5 = keras.layers.Conv1DTranspose(filters=autoencoder_layer5_filters,
                                                       kernel_size=autoencoder_layer5_ks, padding="same",
                                                       activation=LeakyReLU(0.01))(self.dec_layer5)
        self.dec_layer5 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer5)
        self.dec_layer5 = keras.layers.Add()([self.enc_layer5_r, self.dec_layer5])
        self.dec_layer4 = keras.layers.UpSampling1D(autoencoder_layer4_strides)(self.dec_layer5)
        self.dec_layer4 = keras.layers.Cropping1D(cropping=(0, 1))(self.dec_layer4)
        self.dec_layer4 = keras.layers.Conv1DTranspose(filters=autoencoder_layer4_filters,
                                                       kernel_size=autoencoder_layer4_ks, padding="same",
                                                       activation=LeakyReLU(0.01))(self.dec_layer4)
        self.dec_layer4 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer4)
        self.dec_layer4 = keras.layers.Add()([self.enc_layer4_r, self.dec_layer4])
        self.dec_layer3 = keras.layers.UpSampling1D(autoencoder_layer3_strides)(self.dec_layer4)
        self.dec_layer3 = keras.layers.Cropping1D(cropping=1)(self.dec_layer3)
        self.dec_layer3 = keras.layers.Conv1DTranspose(filters=autoencoder_layer3_filters,
                                                       kernel_size=autoencoder_layer3_ks, padding="same",
                                                       activation=LeakyReLU(0.01))(self.dec_layer3)
        self.dec_layer3 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer3)
        self.dec_layer3 = keras.layers.Add()([self.enc_layer3_r, self.dec_layer3])
        self.dec_layer2 = keras.layers.UpSampling1D(autoencoder_layer2_strides)(self.dec_layer3)
        self.dec_layer2 = keras.layers.Cropping1D(cropping=2)(self.dec_layer2)
        self.dec_layer2 = keras.layers.Conv1DTranspose(filters=autoencoder_layer2_filters,
                                                       kernel_size=autoencoder_layer2_ks, padding="same",
                                                       activation=LeakyReLU(0.01))(self.dec_layer2)
        self.dec_layer2 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer2)
        self.dec_layer2 = keras.layers.Add()([self.enc_layer2_r, self.dec_layer2])
        self.dec_layer1 = keras.layers.UpSampling1D(autoencoder_layer1_strides)(self.dec_layer2)
        self.dec_layer1 = keras.layers.Conv1DTranspose(filters=autoencoder_layer1_filters,
                                                       kernel_size=autoencoder_layer1_ks, padding="same",
                                                       activation=LeakyReLU(0.01))(self.dec_layer1)
        self.dec_layer1 = keras.layers.normalization.batch_normalization.BatchNormalization()(self.dec_layer1)
        self.dec_layer1 = keras.layers.Add()([self.enc_layer1_r, self.dec_layer1])
        self.linear_proj = keras.layers.GlobalAveragePooling1D()(self.dec_layer1)
        self.linear_proj = keras.layers.Dense(self.input_size[0], activation=LeakyReLU(0.01))(self.linear_proj)
        self.linear_proj = keras.layers.Dropout(rate=0.1)(self.linear_proj)
        self.linear_proj = keras.layers.Dense(self.input_size[0], activation="linear")(self.linear_proj)
        self.model = Model(inputs=input, outputs=self.linear_proj)
        return self

    def inform(self):
        logging.info("Creating model plot and summary")
        plot_model(self.model, "detrend_autoencoder_resnet.png", show_shapes=True)
        self.model.summary()
        return self

    def compile(self, optimizer, loss, metrics=[]):
        metrics_str = [str(metric) for metric in metrics]
        logging.info("Compiling model with optimizer " + str(optimizer) + ", loss " + str(loss) + " and metrics [" +
            ",".join(metrics_str) + "]")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def prepare_training_data(self, training_dir, train_percent=0.8, test_percent=0.2, training_set_limit=None):
        lc_filenames = [str(file) for file in list(pathlib.Path(training_dir).glob('*_lc.csv'))]
        if training_set_limit is not None:
            lc_filenames = lc_filenames[:training_set_limit]
        lc_filenames.sort()
        lc_filenames = shuffle(lc_filenames)
        dataset_length = len(lc_filenames)
        train_last_index = int(dataset_length * train_percent)
        test_last_index = train_last_index + int(dataset_length * test_percent)
        test_last_index = test_last_index if test_last_index < dataset_length else dataset_length
        return lc_filenames[0:train_last_index], lc_filenames[train_last_index:test_last_index]

    def train(self, training_dir, output_dir, batch_size=20, epochs=20, initial_learning_rate=0.00001,
              dataset_iterations_per_epoch=1, train_percent=0.8,
              test_percent=0.2, training_set_limit=None, inform=False, dry_run=True, zero_epsilon=1e-7,
              gradient_clip_norm=0.01):
        logging.info("Preparing training data with (training_dir," + str(training_dir) +
            ") (train_percent," + str(train_percent) +
            ") (test_percent," + str(test_percent) +
            ") (training_set_limit," + str(training_set_limit) +
            ")")
        train_filenames, test_filenames = self.prepare_training_data(training_dir, train_percent, test_percent,
                                                                     training_set_limit)
        # The optimizer is executed once for every batch, hence optimizer steps per epoch are
        train_dataset_size = len(train_filenames)
        test_dataset_size = len(test_filenames)
        train_dataset_filename = output_dir + "/train_dataset.csv"
        test_dataset_filename = output_dir + "/test_dataset.csv"
        logging.info("Storing train and test file names in " + train_dataset_filename + " and " + test_dataset_filename)
        if os.path.exists(train_dataset_filename):
            os.remove(train_dataset_filename)
        if os.path.exists(test_dataset_filename):
            os.remove(test_dataset_filename)
        with open(output_dir + "/train_dataset.csv", 'w') as file:
            file.write('file_name')
            file.write('\n'.join(train_filenames))
        with open(output_dir + "/test_dataset.csv", 'w') as file:
            file.write('file_name')
            file.write('\n'.join(train_filenames))
        steps_per_epoch = int(dataset_iterations_per_epoch * train_dataset_size // batch_size)
        total_steps = steps_per_epoch * epochs
        learning_rate_decay_steps = total_steps // 100
        learning_rate_decay_steps = learning_rate_decay_steps if learning_rate_decay_steps > epochs else epochs
        logging.info("Initializing optimizer with (initial_learning_rate," + str(initial_learning_rate) +
            ") (learning_rate_decay_steps," + str(learning_rate_decay_steps) +
            ") (gradient_clip_norm," + str(gradient_clip_norm) +
            ")")
        leaning_rate_schedule = ExponentialDecay(initial_learning_rate, decay_steps=learning_rate_decay_steps, decay_rate=0.95)
        optimizer = tf.keras.optimizers.Adam(leaning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             clipnorm=gradient_clip_norm)
        # Autoencoders have a linear output layer and hence, cross entropy is not good (better for softmax
        # classification tasks
        loss = tf.keras.losses.MeanAbsolutePercentageError(reduction=losses_utils.ReductionV2.AUTO)
        self.compile(optimizer, loss, metrics=[FixedMean(), "mean_squared_error",
                                                "mean_absolute_error",
                                                "mean_absolute_percentage_error",
                                                "mean_squared_logarithmic_error",
                                                "cosine_proximity"])
        model_weights_logger = ModelWeightsLogger()
        if inform:
            self.inform()
            model_weights_logger.log_model_weights(self.model)
        if not dry_run:
            logging.info("Initializing training and validation generators with (batch_size," + str(batch_size) +
                ") (self.input_size," + str(self.input_size) +
                ") (zero_epsilon," + str(zero_epsilon) +
                ")")
            training_batch_generator = DetrendModelGenerator(train_filenames, batch_size, self.input_size, zero_epsilon)
            validation_batch_generator = DetrendModelGenerator(test_filenames, batch_size, self.input_size, zero_epsilon)
            csv_log = output_dir + '/training_log.csv'
            logging.info("Initializing csv logger with (csv_log," + str(csv_log) +
                ") (steps_per_epoch," + str(steps_per_epoch) +
                ")")
            csv_logger = BatchAwareCsvLogger(csv_log, steps_per_epoch)
            model_path = output_dir + '/DETREND'
            logging.info("Initializing model checkpoint with (model_path," + str(model_path) +
                ") (steps_per_epoch," + str(steps_per_epoch) +
                ")")
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                verbose=1,
                save_weights_only=True,
                save_freq=steps_per_epoch)
            get_weights_callback = GetWeights(model_weights_logger)
            model_validation_steps = int(test_dataset_size // batch_size)
            logging.info("Initializing training with (epochs," + str(epochs) +
                ") (steps_per_epoch," + str(steps_per_epoch) +
                ") (model_validation_steps," + str(model_validation_steps) +
                ")")
            self.model.fit(x=training_batch_generator,
                           steps_per_epoch=steps_per_epoch,
                           epochs=epochs, verbose=1,
                           validation_data=validation_batch_generator,
                           validation_steps=model_validation_steps,
                           callbacks=[csv_logger, cp_callback, get_weights_callback])
            logging.info("Saving model into " + model_path)
            self.model.save(model_path)
        else:
            logging.info("dry_run was activated and 'training' will not be launched", 'warn')



auto_encoder = AutoEncoder().build()
auto_encoder.train("/mnt/DATA-2/ete6/lcs/", os.getcwd(), 10, 50, inform=True, dry_run=True,
                   training_set_limit=100)
# lc_filenames = [str(file) for file in list(pathlib.Path("/mnt/DATA-2/ete6/lcs/").glob('*_lc.csv'))]
# lc_filenames.sort()
# lc_filenames = shuffle(lc_filenames)
# dataset_length = len(lc_filenames)
# acg = AutoencoderGenerator(lc_filenames, 20, (20610, 7), 1e-7)
# acg.__getitem__(20)
