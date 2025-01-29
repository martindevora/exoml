import os

from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy

from exoml.ml import thread_config
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.ml.callback.get_weights import GetWeights
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.log.get_weights_logger import ModelWeightsLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from exoml.ml.model.base_model import HyperParams
from exoml.santo.SANTO import SANTO
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

lr = 0.001
warmup_lr = lr
batch_size = 1000
epochs = 1
warmup_epochs = 1
# tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
santo = SANTO(transformer_layers=0)
model_output_dir = "/home/administrator/exoml/"
model_dir = f"{model_output_dir}SANTO_model"
model_chk_dir = model_dir + '_chk'
hp = HyperParams(batch_size=batch_size, epochs=epochs, run_eagerly=True, dry_run=False,
                                      train_percent=0.96, validation_percent=0.04,
                                      initial_learning_rate=lr,
                                      #training_set_limit=10,
                                      dropout_rate=0.2, spatial_dropout_rate=0.2, class_loss_weights={0: 1, 1: 1},
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
                                        GetWeights(ModelWeightsLogger()),
                                        WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, warmup_epochs, epochs - warmup_epochs)
                                      ],
                           )
lcs_dir = "/data/scratch/ml/santo/training_data/"
santo.initialize(lcs_dir, model_output_dir, hyperparams=hp)
#santo.load_model(model_chk_dir)
santo.train(lcs_dir, model_output_dir, hyperparams=hp)
