import glob
import os

import numpy
import numpy as np
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy

from exoml.ml import thread_config
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.ml.callback.get_weights import GetWeights
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.log.get_weights_logger import ModelWeightsLogger
from exoml.ml.loss.cross_entropy import binary_focal_loss
from exoml.santo.santo_generator import SantoGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from exoml.ml.model.base_model import HyperParams
from exoml.santo.SANTO import SANTO, weighted_binary_crossentropy_param
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

lr = 0.0001
warmup_lr = lr
batch_size = 64
epochs = 30
warmup_epochs = 1
input_dim = 2000
# tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)

model_output_dir = "/home/administrator/exoml/"
model_dir = f"{model_output_dir}SANTO_model"
model_chk_dir = model_dir + '_chk'
hp = HyperParams(batch_size=batch_size, epochs=epochs, run_eagerly=True, dry_run=False,
                                      train_percent=0.9, validation_percent=0.1,
                                      initial_learning_rate=lr,
                                      #training_set_limit=100,
                                      dropout_rate=0.2, spatial_dropout_rate=0.2,
                                      class_loss_weights=None,
                                      custom_loss=BinaryCrossentropy(),
                                      l1_regularization=0.00,
                                      l2_regularization=0.00,
                                      gradient_clip_norm=2.0,
                                      gradient_clip_value=1.0,
                                      learning_rate_schedule=None,
                                      callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                                        ModelCheckpointCallback(filepath=model_chk_dir,
                                                                            original_path=model_dir,
                                                                            verbose=0, save_weights_only=False,
                                                                            save_best_only=True, monitor='val_p@r99',
                                                                            mode='max', save_freq='epoch',
                                                                            initial_value_threshold=0),
                                        WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, warmup_epochs, epochs - warmup_epochs),
                                        #GetWeights(ModelWeightsLogger())
                                      ],
                           )
lcs_dir = "/data/scratch/ml/santo/training_data/"
santo = SANTO(transformer_layers=0)
santo.initialize(lcs_dir, model_output_dir, hyperparams=hp, input_dim=input_dim)
santo.train(lcs_dir, model_output_dir, hyperparams=hp, input_dim=input_dim)
