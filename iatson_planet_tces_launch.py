import pydevd_pycharm

#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os

from keras.losses import BinaryCrossentropy

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from iatson.IATSON_planet import IATSON_planet
from ml.loss.cross_entropy import f1_loss
from ml.model.base_model import HyperParams

training_dir = '/data/scratch/ml/'
model_dir = './'
hp = HyperParams(batch_size=100,
            epochs=75,
            initial_learning_rate=0.01,
            gradient_clip_norm=0.1,
            gradient_clip_value=0.5,
            dry_run=False,
            cores=0,
            balance_class_sampling=[10, 10, 5, 1, 1, 2.5, 2.5],
            learning_rate_decay=0.98,
            early_stopping_delta=0,
            early_stopping_patience=0,
            custom_loss=BinaryCrossentropy(),
            run_eagerly=False,
            dropout_rate=0.2,
            white_noise_std=0.01)
iatson = IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'tce', 4: 'none', 5: 'EB', 6: 'bckEB'},
                       ['planet_transit', 'planet', 'fp', 'tce', 'none', 'EB', 'bckEB'],
                       {'fp': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'none': [0],
                        'EB': [0], 'bckEB': [0]}, hp)\
    .build(use_transformers=False, transformer_blocks=3, transformer_heads=3).inform(model_dir)
#iatson.load_model("./IATSON_planet_model_bck")
iatson.train(training_dir, model_dir, hp)
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, "./IATSON_planet_model_bck", batch_size, os.cpu_count() - 1)
