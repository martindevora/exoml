import os

from iatson.IATSON import IATSON
import tensorflow as tf

from ml import thread_config
from ml.metrics import auc
from ml.metrics.auc import ConfusionMatrixStatisticsCallback

tf.config.run_functions_eagerly(True)
thread_config.setup_threads(1)

training_dir = '/mnt/DATA-2/ete6'
batch_size = 10
epochs = 50
model_dir = './'
iatson = IATSON().build()#.inform(model_dir)
iatson.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=0.01, gradient_clip_norm=0.1,
             dry_run=False, cores=os.cpu_count() - 1, training_set_limit=30,
             callbacks=[ConfusionMatrixStatisticsCallback(['backeb', 'eb', 'planet', 'none'])])
iatson = IATSON().load_model(model_dir)
iatson.predict_test_set(training_dir, './test_dataset.csv', batch_size)
