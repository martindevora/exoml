import os

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from detrend.DETREND import DETREND
from ml.model.base_model import HyperParams

training_dir = '/mnt/DATA-2/ete6/lcs/'
model_dir = './'
hp = HyperParams(batch_size=10,
            epochs=200,
            initial_learning_rate=0.00001,
            gradient_clip_norm=0.1,
            gradient_clip_value=0.5,
            dry_run=False,
            cores=0,
            learning_rate_decay=0.98,
            early_stopping_delta=0,
            early_stopping_patience=0,
            custom_loss=None,
            run_eagerly=True,
            dropout_rate=0.2,
            white_noise_std=0.01,
            training_set_limit=100,
            metrics=["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                     "mean_squared_logarithmic_error", "cosine_proximity"])
detrend = DETREND(hp).build().inform(model_dir)
#auto_encoder.inform("DETREND.png")
detrend.train(training_dir, model_dir, hp)
