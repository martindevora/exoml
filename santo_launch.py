import os

from exoml.ml import thread_config
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.ml.callback.get_weights import GetWeights
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.log.get_weights_logger import ModelWeightsLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from exoml.ml.model.base_model import HyperParams
from exoml.santo.SANTO import SANTO
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

lr = 0.001
warmup_lr = 1e-7
batch_size = 1000
epochs = 10
steps_per_epoch = 386 * 0.9
# tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
SANTO(transformer_layers=1).train("/data/scratch/ml/santo/training_data/", "/data/scratch/ml/santo/",
              hyperparams=HyperParams(batch_size=batch_size, epochs=epochs, run_eagerly=True, dry_run=False,
                                      initial_learning_rate=lr,
                                      dropout_rate=0.2, spatial_dropout_rate=0.2, class_loss_weights={0: 1, 1:1},
                                      l1_regularization=0.0001,
                                      l2_regularization=0.0001,
                                      gradient_clip_norm=0.01,
                                      gradient_clip_value=0.05,
                                      # training_set_limit=3,
                                      learning_rate_schedule=None,
                                      callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                                        ModelCheckpointCallback(filepath="./SANTO_model_chk",
                                                                            original_path='/SANTO_model',
                                                                            verbose=0, save_weights_only=False,
                                                                            save_best_only=True, monitor='val_p@r99',
                                                                            mode='max', save_freq='epoch',
                                                                            initial_value_threshold=0),
                                        WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, 1, epochs - 1),
                                        GetWeights(ModelWeightsLogger())
                                      ],

                                      )
              )
