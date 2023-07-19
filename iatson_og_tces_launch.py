#import pydevd_pycharm

#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import shutil
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow_addons as tfa

from iatson.IATSON_og import IATSON_og
from ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from ml.model.base_model import HyperParams

training_dir = '/data/scratch/ml/'
model_dir = './'
batch_size = 200 #100
lr = 0.001 #0.0001
warmup_lr = 1e-7
epochs = 400
cores = 10
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
                 epochs=epochs, #125
                 initial_learning_rate=warmup_lr, #lr,
                 learning_rate_schedule=learning_rate_schedule, #learning_rate_schedule,
                 lr_progression=1,
                 gradient_clip_norm=None, #0.1, #0.1,
                 gradient_clip_value=None, #0.5, #0.5,
                 dry_run=False,
                 cores=cores,
                 balance_class_sampling=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 learning_rate_decay=0.98,
                 early_stopping_delta=0,
                 early_stopping_patience=10,
                 class_loss_weights=None, #None, #'auto', #{0: 1, 1: 10},
                 custom_loss=None, #BinaryCrossentropy(),
                 run_eagerly=False,
                 dropout_rate=0.1,
                 dropout_max_rate=0.1,
                 spatial_dropout_rate=0.3,
                 white_noise_std=0.001, #0.001, #0.001, #0.001, #None,
                 l1_regularization=0.0005, #0.001, #0.00001, #0.00001, #None,
                 l2_regularization=0.0005, #0.001, #0.00001, #0.00001, #None,
                 stochastic_weight_average_wait=0, #int(0.75 * epochs),
                 callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                            WarmUpAndLinDecreaseCallback(warmup_lr, lr, warmup_lr, epochs // 20, epochs - epochs // 20),
                            ModelCheckpoint(filepath="./IATSON_og_model_chk", verbose=0, save_weights_only=False, save_best_only=True, monitor='val_r@p99', mode='max', save_freq='epoch', initial_value_threshold=0.85)]
)
shutil.rmtree("./IATSON_og_model", ignore_errors=True)
iatson = IATSON_og({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                        6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                        10: 'none', 11: 'EB', 12: 'bckEB'},
                       ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                        'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB'],
                       {'fp': [0], 'fa': [0], 'planet': [0], 'planet_transit': [0], 'tce': [0], 'tce_centroids_offset': [0],
                        'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [1], 'tce_odd_even': [0], 'none': [0],
                        'EB': [0], 'bckEB': [0]}, hp, mode="q1q17")\
    .build(use_transformers=True, transformer_blocks=2, transformer_heads=4).inform(model_dir)
#iatson.load_model("./IATSON_planet_model_rp95_0.7")
iatson.train(training_dir, model_dir, hp)
#iatson.train_cv(training_dir, model_dir, hp)
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
#iatson = iatson.load_model("/usr/bin/exoml/IATSON_planet_model/")
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
