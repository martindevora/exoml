import pandas as pd
import numpy as np
import tensorflow as tf
import time
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils import shuffle

from ml.layers.transformer import Transformer
from ml.model.base_model import HyperParams


class SANTO:
    """
    Self-Attention Neural Network for Transiting Objects
    """
    def __init__(self) -> None:
        super().__init__()

    def loss_function(self, loss_holder, target, pred):
        #Custom loss function to mask the padding tokens
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = loss_holder(target, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train(self, lcs_dir, injected_objects_file, model_dir, n_epochs,
              hyperparams: HyperParams, input_dim=(500, 2), batch_size=1000):
        tf.keras.backend.clear_session()
        output_dim = len(input_dim)
        checkpoint_path = f"{model_dir}_chk"
        transformer = Transformer(num_heads=4, key_dim=input_dim, ffn_units=input_dim, output_dim=output_dim,
                                  stack_depth=2, dropout_rate=hyperparams.dropout_rate)
        # TODO this loss function is probably not well suited for our purpose as we don't want an exclusive output, but
        #  a linear one
        # Define a categorical cross entropy loss
        loss_holder = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        # Define a metric to store the mean loss of every epoch
        train_loss = tf.keras.metrics.Mean(name="loss")
        # Define a metric to save the accuracy in every epoch
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        # Define a metric to store the mean loss of every epoch
        train_precision = tf.keras.metrics.Precision(name="precision")
        # Define a metric to store the mean loss of every epoch
        train_recall = tf.keras.metrics.Recall(name="recall")
        # Define a metric to store the mean loss of every epoch
        train_pr99 = tf.keras.metrics.PrecisionAtRecall(recall=0.99, name="p@r99")
        # Create the Adam optimizer (Vaswani et al., 2017) with learning rate decay from Schedule
        leaning_rate = CustomSchedule(input_dim)
        optimizer = tf.keras.optimizers.Adam(leaning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        # Create the Checkpoint
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Last checkpoint restored.")

        ''' Train the transformer model for n_epochs using the data generator dataset'''
        injected_objects_df = pd.read_csv(injected_objects_file)
        injected_objects_df = injected_objects_df.sort_values(by=["object_id"], ascending=True)
        injected_objects_df_unique = injected_objects_df.drop_duplicates(subset=["object_id"], keep='last')
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        pr99s = []
        for epoch in range(n_epochs):
            print("Epoch {}".format(epoch + 1))
            start = time.time()
            # In every epoch reset the loss and accuracy calculations
            train_loss.reset_states()
            train_accuracy.reset_states()
            for index, row in injected_objects_df_unique.iterrows():
                input_lc_df = None
                for object_index, object_row in injected_objects_df[injected_objects_df['object_id'] == row['object_id']].iterrows():
                    lc_df = pd.read_csv(f"{lcs_dir}/{row['object_id']}_{round(object_row['period'], 2)}_lc_short.csv",
                                        usecols=['#time', 'flux', 'planet_model', 'eb_model', 'bckeb_model'])
                    if input_lc_df is None:
                        input_lc_df = lc_df
                    else:
                        input_lc_df['planet_model_flux'] = input_lc_df['planet_model_flux'] - 1 + lc_df['planet_model_flux']
                        input_lc_df['eb_model'] = input_lc_df['eb_model'] - 1 + lc_df['eb_model']
                        input_lc_df['bckeb_model'] = input_lc_df['bckeb_model'] - 1 + lc_df['bckeb_model']
                input_lc_df.dropna(inplace=True)
                inputs = np.zeros((batch_size, input_dim[0], input_dim[1]))
                outputs = np.zeros((batch_size, input_dim[0]))
                time_arr = np.ones(len(input_dim) // 2) + input_lc_df['#time'].to_numpy() + np.ones(input_dim[0] // 2)
                # TODO encode time??
                flux = np.ones(len(input_dim) // 2) + input_lc_df['flux'].to_numpy() + np.ones(input_dim[0] // 2)
                flux = flux / 2
                input_values = np.column_stack((time_arr, flux))
                model_values = ((input_lc_df['planet_model_flux'].to_numpy() < 1) |
                                (input_lc_df['eb_model'].to_numpy() < 1) |
                                (input_lc_df['bckeb_model'].to_numpy() < 0))
                input_windows = sliding_window_view(input_values, window_shape=input_dim[0])
                flux_predictions = transformer(input_windows, model_values, True)
                loss = self.loss_function(model_values, flux_predictions, loss_holder)
                train_loss(loss)
                train_accuracy(input_windows, flux_predictions)
                train_precision(input_windows, flux_predictions)
                train_recall(input_windows, flux_predictions)
                train_pr99(input_windows, flux_predictions)
                losses.append(train_loss.result())
                accuracies.append(train_accuracy.result())
                precisions.append(train_accuracy.result())
                recalls.append(train_accuracy.result())
                pr99s.append(train_accuracy.result())
                print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(epoch + 1, index, train_loss.result(),
                                                                             train_accuracy.result()))
            # Checkpoint the model on every epoch
            ckpt_save_path = ckpt_manager.save()
            print("Saving checkpoint for epoch {} in {}".format(epoch + 1, ckpt_save_path))
            print("Time for 1 epoch: {} secs\n".format(time.time() - start))
        return losses, accuracies


class AutoencoderGenerator(tf.keras.utils.Sequence):
    def __init__(self, lc_filenames, batch_size, input_size):
        self.lc_filenames = lc_filenames
        self.batch_size = batch_size
        self.input_size = input_size

    def __len__(self):
        return (np.ceil(len(self.lc_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_filenames = self.lc_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        filenames_shuffled = shuffle(batch_filenames)
        batch_data_array = np.empty((len(filenames_shuffled), self.input_size[0], self.input_size[1]))
        batch_data_values = np.empty((len(filenames_shuffled), (self.input_size, 4)))
        i = 0
        for file in filenames_shuffled:
            input_df = pd.read_csv(file, usecols=['flux', 'flux_err'], low_memory=True)
            values_df = pd.read_csv(file, usecols=['eb_model', 'bckeb_model', 'planet_model'],
                                    low_memory=True)
            batch_data_array[i] = input_df.to_numpy()
            eb_arr = values_df['eb_model'].to_numpy()
            bckeb_arr = values_df['bckeb_model'].to_numpy()
            planet_arr = values_df['planet_model'].to_numpy()
            eb_labels_args = np.argwhere(eb_arr < 1).flatten()
            bckeb_labels_args = np.argwhere(bckeb_arr < 1).flatten()
            planet_labels_args = np.argwhere(planet_arr < 1).flatten()
            nothing_labels_args = np.argwhere((eb_arr == 1) & (bckeb_arr == 1) & (planet_arr == 1)).flatten()
            eb_label = np.zeros(self.input_size[0])
            eb_label[eb_labels_args] = 1
            bckeb_label = np.zeros(self.input_size[0])
            bckeb_label[bckeb_labels_args] = 1
            planet_label = np.zeros(self.input_size[0])
            planet_label[planet_labels_args] = 1
            nothing_label = np.zeros(self.input_size[0])
            nothing_label[nothing_labels_args] = 1
            batch_data_values[i] = np.transpose([nothing_label, eb_label, bckeb_label, planet_label])
            batch_data_values[i] = values_df['model'].to_numpy()
            i = i + 1
        return batch_data_array, batch_data_values

    def __prepare_input_data(self, input_df):
        time = input_df["#time"].to_numpy()
        dif = time[1:] - time[:-1]
        jumps = np.where(np.abs(dif) > 0.25)[0]
        jumps = np.append(jumps, len(input_df))
        previous_jump_index = 0
        for jumpIndex in jumps:
            token = input_df["centroid_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_x"][previous_jump_index:jumpIndex] = np.tanh(token - np.nanmedian(token))
            token = input_df["centroid_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["centroid_y"][previous_jump_index:jumpIndex] = np.tanh(token - np.nanmedian(token))
            token = input_df["motion_x"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_x"][previous_jump_index:jumpIndex] = np.tanh(token - np.nanmedian(token))
            token = input_df["motion_y"][previous_jump_index:jumpIndex].to_numpy()
            input_df["motion_y"][previous_jump_index:jumpIndex] = np.tanh(token - np.nanmedian(token))
            token = input_df["bck_flux"][previous_jump_index:jumpIndex].to_numpy()
            input_df["bck_flux"][previous_jump_index:jumpIndex] = np.tanh(token - np.nanmedian(token))
            previous_jump_index = jumpIndex
        return input_df.fillna(0)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        warmup_steps = tf.cast(tf.convert_to_tensor(self.warmup_steps, dtype=tf.int64, name="warmup_steps"), tf.float32)
        arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
        arg2 = tf.cast(step, tf.float32) * tf.math.pow(warmup_steps, -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
