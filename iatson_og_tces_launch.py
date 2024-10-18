#import pydevd_pycharm

#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy

from exoml.iatson.IATSON_og import IATSON_og
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.ml.callback.get_weights import GetWeights
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.log.get_weights_logger import ModelWeightsLogger
from exoml.ml.model.base_model import HyperParams

training_dir = '/data/scratch/ml/'
model_dir = '/home/administrator/exoml/'
batch_size = 200 #100
lr = 0.001 #0.0001
warmup_lr = 1e-7
epochs = 400
cores = 15
steps_per_epoch = 148
#learning_rate_schedule = ExponentialRescaleDecay(lr,
                                                 # decay_steps=20,
                                                 # decay_rate=0.98,
                                                 # restore_steps=200,
                                                 # restore_rate=1.5,
                                                 # staircase=True)
#learning_rate_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=lr,
#    maximal_learning_rate=warmup_lr,
#    scale_fn=lambda x: 1/(2.**(x-1)),
#    step_size=5 * steps_per_epoch
#)
#learning_rate_schedule = LinearWithWarmupDecaySchedule(1e-7,
#                                                       lr,
#                                                       1e-6,
#                                                       1000,
#                                                       baseline_steps=40000,
#                                                       stair_steps=0)
learning_rate_schedule = None
hp = HyperParams(batch_size=batch_size,
                 epochs=epochs,  #125
                 initial_learning_rate=lr,  #lr,
                 learning_rate_schedule=learning_rate_schedule,  #learning_rate_schedule,
                 lr_progression=0.9,
                 gradient_clip_norm=0.01,  # 0.1,
                 gradient_clip_value=0.05,  #0.5, #0.5,
                 dry_run=False,
                 cores=cores,
                 balance_class_sampling=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 learning_rate_decay=0.98,
                 early_stopping_delta=0,
                 early_stopping_patience=10,
                 class_loss_weights=None,  #None, #'auto', #{0: 1, 1: 10},
                 custom_loss=BinaryCrossentropy(),
                 run_eagerly=True,
                 dropout_rate=0,
                 dropout_max_rate=0,
                 spatial_dropout_rate=0,
                 white_noise_std=0,  #0.001, #0.001, #0.001, #None,
                 l1_regularization=0,  #0.001, #0.00001, #0.00001, #None,
                 l2_regularization=0,  #0.001, #0.00001, #0.00001, #None,
                 stochastic_weight_average_wait=0, #int(0.75 * epochs),  #int(0.75 * epochs),
                 callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                            WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, 1, epochs - 1),
                            ModelCheckpointCallback(filepath="./IATSON_og_model_chk",
                                                                original_path='/IATSON_og_model',
                                                                verbose=0, save_weights_only=False,
                                                                save_best_only=True, monitor='val_r@p99',
                                                                mode='max', save_freq='epoch',
                                                                initial_value_threshold=0.8),
                            GetWeights(ModelWeightsLogger())]
                 )
shutil.rmtree(model_dir + "/IATSON_og_model", ignore_errors=True)
iatson = IATSON_og({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                        6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                        10: 'none', 11: 'EB', 12: 'bckEB'},
                       ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                        'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB'],
                       {'fp': [1], 'fa': [1], 'planet': [1], 'planet_transit': [1], 'tce': [1], 'tce_centroids_offset': [1],
                        'tce_source_offset': [1], 'tce_secondary': [1], 'tce_og': [0], 'tce_odd_even': [1], 'none': [1],
                        'EB': [1], 'bckEB': [1]}, hp, mode="q1q17")\
    .build(use_transformers=True, transformer_blocks=1, transformer_heads=2).inform(model_dir)
#iatson.load_model("./IATSON_planet_model_rp95_0.7")
iatson.train(training_dir, model_dir, hp)
#iatson.train_cv(training_dir, model_dir, hp)
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
#iatson = iatson.load_model("/usr/bin/exoml/IATSON_planet_model/")
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
