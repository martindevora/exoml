import logging
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import LeakyReLU
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from keras.utils import losses_utils
from sklearn.utils import shuffle

from ml.callback.batch_aware_csv_logger import BatchAwareCsvLogger
from ml.callback.get_weights import GetWeights
from iatson.iatson_generator import IatsonModelGenerator
from ml.log.get_weights_logger import ModelWeightsLogger
from ml.model.base_model import BaseModel


class IATSON(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.input_size = 8

    def _get_flux_conv_branch(self, name):
        # (time, flux, [detrended_flux1 ... detrended_flux5], flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        leaky_relu_alpha = 0.01
        flux_input = keras.Input(shape=(2500, 8), name=name)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_input)
        flux_branch = keras.layers.SpatialDropout1D(rate=0.2)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=128, kernel_size=9, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=10, strides=6)(flux_branch)
        flux_branch = keras.layers.Dropout(rate=0.1)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=4, strides=3)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=32, kernel_size=5, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(flux_branch)
        flux_branch = keras.layers.Conv1D(filters=16, kernel_size=3, padding="same",
                                          activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.MaxPooling1D(pool_size=2, strides=2)(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        return flux_input, flux_branch

    def _get_flux_model_branch(self):
        leaky_relu_alpha = 0.01
        flux_input, flux_branch = self._get_flux_conv_branch("global_flux_branch")
        flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(flux_branch)
        flux_branch = keras.layers.Dropout(rate=0.1)(flux_branch)
        flux_branch = keras.layers.Dense(16, activation='linear')(flux_branch)
        flux_branch = keras.layers.LayerNormalization()(flux_branch)
        flux_branch = keras.Model(inputs=[flux_input], outputs=flux_branch)
        return flux_branch

    def _get_focus_flux_branch(self, name):
        leaky_relu_alpha = 0.01
        focus_flux_input = keras.Input(shape=(500, 13), name=name)
        flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_input)
        # (time, flux, detrended_flux1... detrended_flux5, flux_model, centroidx, centroidy, motionx, motiony, bck)
        # flux model by transit params and stellar params
        focus_flux_branch = keras.layers.SpatialDropout1D(rate=0.2)(focus_flux_input)
        focus_flux_branch = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=5, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=0.1)(focus_flux_branch)
        focus_flux_branch = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same",
                                                activation=LeakyReLU(leaky_relu_alpha), use_bias=True)(
            focus_flux_branch)
        focus_flux_branch = keras.layers.normalization.batch_normalization.BatchNormalization()(focus_flux_branch)
        focus_flux_branch = keras.layers.MaxPooling1D(pool_size=3, strides=3)(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dropout(rate=0.1)(focus_flux_branch)
        focus_flux_branch = keras.layers.LayerNormalization()(focus_flux_branch)
        return focus_flux_input, focus_flux_branch

    def _get_focus_flux_model_branch(self):
        leaky_relu_alpha = 0.01
        odd_flux_input, odd_flux_branch = self._get_focus_flux_branch("focus_odd_flux_branch")
        even_flux_input, even_flux_branch = self._get_focus_flux_branch("focus_even_flux_branch")
        harmonic_odd_flux_input, harmonic_odd_flux_branch = self._get_focus_flux_branch("focus_harmonic_odd_flux_branch")
        harmonic_even_flux_input, harmonic_even_flux_branch = self._get_focus_flux_branch("focus_harmonic_even_flux_branch")
        subharmonic_odd_flux_input, subharmonic_odd_flux_branch = self._get_focus_flux_branch(
            "focus_subharmonic_odd_flux_branch")
        subharmonic_even_flux_input, subharmonic_even_flux_branch = self._get_focus_flux_branch(
            "focus_subharmonic_even_flux_branch")
        odd_flux_branch = keras.layers.Add()(
            [odd_flux_branch, harmonic_odd_flux_branch, subharmonic_odd_flux_branch])
        even_flux_branch = keras.layers.Add()(
            [even_flux_branch, harmonic_even_flux_branch, subharmonic_even_flux_branch])
        odd_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(odd_flux_branch)
        odd_flux_branch = keras.layers.Dropout(rate=0.1)(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha))(even_flux_branch)
        even_flux_branch = keras.layers.Dropout(rate=0.1)(even_flux_branch)
        odd_flux_branch = keras.layers.Dense(32, activation='linear')(odd_flux_branch)
        even_flux_branch = keras.layers.Dense(32, activation='linear')(even_flux_branch)
        odd_flux_branch = keras.layers.LayerNormalization()(odd_flux_branch)
        even_flux_branch = keras.layers.LayerNormalization()(even_flux_branch)
        focus_flux_branch = keras.layers.Add()([odd_flux_branch, even_flux_branch])
        focus_flux_branch = keras.layers.Dense(24, activation=LeakyReLU(leaky_relu_alpha))(focus_flux_branch)
        focus_flux_branch = keras.layers.Dense(16, activation='linear')(focus_flux_branch)
        input = [even_flux_input, odd_flux_input, subharmonic_even_flux_input, subharmonic_odd_flux_input,
                 harmonic_even_flux_input, harmonic_odd_flux_input]
        focus_flux_branch = keras.Model(inputs=input, outputs=focus_flux_branch)
        return focus_flux_branch

    def _get_singletransit_tpf_model(self):
        video_image_width = 13
        video_image_height = 13
        video_image_channels = 1
        sequences_per_video = 100
        tpf_model_input = keras.Input(
            shape=(video_image_height, video_image_width, sequences_per_video, video_image_channels),
            name="tpf_input")
        tpf_model = keras.layers.SpatialDropout3D(rate=0.3)(tpf_model_input)
        tpf_model = keras.layers.Conv3D(filters=100, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.2)(tpf_model)
        tpf_model = keras.layers.Conv3D(filters=200, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation="relu")(
            tpf_model)
        tpf_model = keras.layers.SpatialDropout3D(rate=0.1)(tpf_model)
        tpf_model = keras.layers.MaxPooling3D(pool_size=(5, 5, 10), strides=(3, 3, 6), padding='same')(tpf_model)
        tpf_model = keras.layers.Dense(200, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(100, activation="relu")(tpf_model)
        tpf_model = keras.layers.Dense(20, activation="relu")(tpf_model)
        tpf_model = keras.layers.Flatten()(tpf_model)
        return keras.Model(inputs=tpf_model_input, outputs=tpf_model)

    def _get_singletransit_motion_centroids_model(self):
        mc_input = keras.Input(
            shape=(100, 4),
            name="motion_centroids_input")
        mc_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                       padding='same')(mc_input)
        mc_model = keras.layers.SpatialDropout1D(rate=0.3)(mc_model)
        mc_model = keras.layers.Conv1D(filters=100, kernel_size=5, strides=5, activation="relu", use_bias=True,
                                       padding='same')(mc_model)
        mc_model = keras.layers.SpatialDropout1D(rate=0.2)(mc_model)
        mc_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(mc_model)
        mc_model = keras.layers.Dense(50, activation="relu")(mc_model)
        mc_model = keras.layers.Dense(20, activation="relu")(mc_model)
        mc_model = keras.layers.Flatten()(mc_model)
        return keras.Model(inputs=mc_input, outputs=mc_model)

    def _get_singletransit_bckflux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="bck_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def _get_singletransit_flux_model(self):
        bck_input = keras.Input(shape=(100, 1), name="flux_input")
        bck_model = keras.layers.Conv1D(filters=25, kernel_size=2, strides=2, activation="relu", use_bias=True,
                                        padding='same')(bck_input)
        bck_model = keras.layers.SpatialDropout1D(rate=0.3)(bck_model)
        bck_model = keras.layers.Conv1D(filters=50, kernel_size=3, strides=3, activation="relu", use_bias=True,
                                        padding='same')(bck_model)
        bck_model = keras.layers.SpatialDropout1D(rate=0.2)(bck_model)
        bck_model = keras.layers.MaxPooling1D(pool_size=5, strides=3, padding='same')(bck_model)
        bck_model = keras.layers.Dense(50, activation="relu")(bck_model)
        bck_model = keras.layers.Dense(10, activation="relu")(bck_model)
        bck_model = keras.layers.Flatten()(bck_model)
        return keras.Model(inputs=bck_input, outputs=bck_model)

    def _get_single_transit_model(self):
        tpf_branch = self._get_singletransit_tpf_model()
        mc_branch = self._get_singletransit_motion_centroids_model()
        bck_branch = self._get_singletransit_bckflux_model()
        flux_branch = self._get_singletransit_flux_model()
        final_branch = keras.layers.concatenate(
            [tpf_branch.output, mc_branch.output, bck_branch.output, flux_branch.output], axis=1)
        final_branch = keras.layers.Dense(64, activation="relu", name="final-dense1")(final_branch)
        final_branch = keras.layers.Dense(32, activation="relu", name="final-dense2")(final_branch)
        final_branch = keras.layers.Dense(1, activation="softmax", name="final-dense-softmax")(final_branch)
        inputs = tpf_branch.inputs + mc_branch.inputs + bck_branch.inputs + flux_branch.inputs
        model = keras.Model(inputs=inputs, outputs=final_branch, name="mnist_model")
        keras.utils.vis_utils.plot_model(model, "IATSON_model.png", show_shapes=True)

    def build(self):
        leaky_relu_alpha = 0.01
        stellar_model_input = keras.Input(shape=(11, 1), name="stellar_model")
        stellar_model_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="stellar-first")(stellar_model_input)
        stellar_model_branch = keras.layers.Dropout(rate=0.1, name="stellar-first-dropout-0.1")(stellar_model_branch)
        stellar_model_branch = keras.layers.Dense(16, activation="relu", name="stellar-refinement")(
            stellar_model_branch)
        flux_model_branch = self._get_flux_model_branch()
        focus_flux_model_branch = self._get_focus_flux_model_branch()
        final_branch = keras.layers.Concatenate(axis=1)(
            [stellar_model_branch, flux_model_branch.output, focus_flux_model_branch.output])
        final_branch = keras.layers.Dense(64, activation=LeakyReLU(leaky_relu_alpha), name="final-dense1")(final_branch)
        final_branch = keras.layers.Dense(16, activation=LeakyReLU(leaky_relu_alpha), name="final-dense2")(final_branch)
        final_branch = keras.layers.Dense(4, activation="softmax", name="final-dense-softmax")(final_branch)
        inputs = [stellar_model_input] + flux_model_branch.inputs + focus_flux_model_branch.inputs
        self.set_model(keras.Model(inputs=inputs, outputs=final_branch, name="IATSON"))
        return self

    def load_training_set(self, **kwargs):
        training_dir = kwargs.get('training_dir')
        injected_objects_df = pd.read_csv(training_dir + "/injected_objects.csv", index_col=None)
        injected_objects_df = shuffle(injected_objects_df)
        injected_objects_df.reset_index(inplace=True, drop=True)
        return injected_objects_df

    def train(self, training_dir, output_dir, batch_size=20, epochs=20, initial_learning_rate=0.01,
              dataset_iterations_per_epoch=1, train_percent=0.8,
              test_percent=0.2, training_set_limit=None, dry_run=True, zero_epsilon=1e-7,
              gradient_clip_norm=0.01):
        logging.info("Preparing training data with (training_dir," + str(training_dir) +
                     ") (train_percent," + str(train_percent) +
                     ") (test_percent," + str(test_percent) +
                     ") (training_set_limit," + str(training_set_limit) +
                     ")")
        train_filenames, test_filenames = self.prepare_training_data(training_dir, output_dir, train_percent,
                                                                     test_percent,
                                                                     training_set_limit)
        # The optimizer is executed once for every batch, hence optimizer steps per epoch are
        train_dataset_size = len(train_filenames)
        test_dataset_size = len(test_filenames)
        steps_per_epoch = int(dataset_iterations_per_epoch * train_dataset_size // batch_size)
        total_steps = steps_per_epoch * epochs
        learning_rate_decay_steps = total_steps // 100
        learning_rate_decay_steps = learning_rate_decay_steps if learning_rate_decay_steps > epochs else epochs
        logging.info("Initializing optimizer with (initial_learning_rate," + str(initial_learning_rate) +
                     ") (learning_rate_decay_steps," + str(learning_rate_decay_steps) +
                     ") (gradient_clip_norm," + str(gradient_clip_norm) +
                     ")")
        leaning_rate_schedule = ExponentialDecay(initial_learning_rate, decay_steps=learning_rate_decay_steps,
                                                 decay_rate=0.95)
        optimizer = tf.keras.optimizers.Adam(leaning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
                                             clipnorm=gradient_clip_norm)
        # We don't use SparseCategoricalCrossentropy because our targets are one-hot encoded
        loss = CategoricalCrossentropy(reduction=losses_utils.ReductionV2.AUTO)
        accuracy = CategoricalAccuracy()
        self.compile(optimizer, loss, metrics=[accuracy])
        model_weights_logger = ModelWeightsLogger()
        if not dry_run:
            logging.info("Initializing training and validation generators with (batch_size," + str(batch_size) +
                         ") (self.input_size," + str(self.input_size) +
                         ") (zero_epsilon," + str(zero_epsilon) +
                         ")")
            lcs_dir = training_dir + '/lcs/'
            training_batch_generator = IatsonModelGenerator(train_filenames, lcs_dir, batch_size, self.input_size,
                                                            zero_epsilon)
            validation_batch_generator = IatsonModelGenerator(test_filenames, lcs_dir, batch_size, self.input_size,
                                                              zero_epsilon)
            csv_log = output_dir + '/training_log.csv'
            logging.info("Initializing csv logger with (csv_log," + str(csv_log) +
                         ") (steps_per_epoch," + str(steps_per_epoch) +
                         ")")
            csv_logger = BatchAwareCsvLogger(csv_log, steps_per_epoch)
            model_path = output_dir + '/IATSON'
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
            logging.warning("dry_run was activated and 'training' will not be launched")

iatson = IATSON().build().inform("IATSON.png")
iatson.train('/mnt/DATA-2/ete6/', './', 10, 50, dry_run=True)
