import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from exoml.ml.model.base_model import HyperParams
from exoml.santo.SANTO import SANTO
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


hp = HyperParams(dropout_rate=0.1)
SANTO().train("/data/scratch/ml/santo/training_data/", "/data/scratch/ml/santo",
              hyperparams=HyperParams(batch_size=10000, epochs=10, run_eagerly=True, initial_learning_rate=0.00001,
                                      dropout_rate=0.2, spatial_dropout_rate=0.2, class_loss_weights={0: 20}))
