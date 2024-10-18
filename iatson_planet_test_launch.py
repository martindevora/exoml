#import pydevd_pycharm

#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
#from tensorflow.python.client import device_lib

#device_lib.list_local_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import shutil
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
from exoml.ml.loss.cross_entropy import binary_focal_loss
#import tensorflow_addons as tfa

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.iatson.IATSON_planet import IATSON_planet
from exoml.ml.learning_rate.schedulers import ExponentialRescaleDecay
from exoml.ml.loss.cross_entropy import fb_loss, dice_bce_loss, dice_loss
from exoml.ml.model.base_model import HyperParams

training_dir = '/data/scratch/ml/'
model_dir = './'
batch_size = 100 #100
lr = 0.0003 #0.0003
warmup_lr = 1e-6
epochs = 400 #400
cores = 20
#steps_per_epoch = 148
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
                 epochs=epochs,
                 train_percent=0.9,
                 initial_learning_rate=lr, #lr,
                 learning_rate_schedule=None,
                 lr_progression=0.9, #0.9
                 gradient_clip_norm=0.01, #0.1,
                 gradient_clip_value=0.05, #0.5,
                 dry_run=False,
                 cores=cores,
                 balance_class_sampling=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 learning_rate_decay=0.98,
                 early_stopping_delta=0,
                 early_stopping_patience=10,
                 class_loss_weights=None, #{0: 1, 1: 6}, #None,
                 custom_loss=BinaryCrossentropy(), #binary_focal_loss,
                 run_eagerly=False,
                 dropout_rate=0.4, # 0.15
                 dropout_max_rate=0.4, # 0.15
                 spatial_dropout_rate=0.3, # 0.3
                 white_noise_std=0.002, #0.001,
                 white_noise_layer_std=0.000,
                 numerical_white_noise_std=0.00,
                 l1_regularization=0.00, #0.0001 #0.00005 0.00007-4,8
                 l2_regularization=0.00, #0.00002,
                 l1_regularization_conv=0.00000,
                 l2_regularization_conv=0.000000,
                 stochastic_weight_average_wait=int(0.75 * epochs),
                 callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                            ModelCheckpointCallback(filepath="./IATSON_planet_model_chk",
                                                                original_path='/IATSON_planet_model',
                                                                verbose=0, save_weights_only=False,
                                                                save_best_only=True, monitor='val_r@p99',
                                                                mode='max', save_freq='epoch',
                                                                initial_value_threshold=0.68),
                            WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, 1, epochs - 1)]
)
shutil.rmtree("./option2/IATSON_planet_model", ignore_errors=True)

iatson = IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                        6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                        10: 'none', 11: 'EB', 12: 'bckEB', 13: 'tce_candidate'},
                       ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                        'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB', 'tce_candidate'],
                       {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'tce_centroids_offset': [0],
                        'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0], 'tce_odd_even': [0], 'none': [0],
                        'EB': [0], 'bckEB': [0], 'tce_candidate': [0]}, hp,
                        mode=['q1q17', 'ete6', 'with_radius', 'no_val']) #, 'with_tce_candidates'])

#with iatson.train_strategy(["/physical_device:GPU:1"]).scope():
iatson.build(use_transformers=False, normalization_mode='layer_norm', transformer_blocks=1, transformer_heads=2).inform(model_dir)
#df = iatson.load_training_set(training_dir=training_dir)
#kepler_df = df.loc[df['object_id'].str.startswith("KIC")]
#tess_df = df.loc[df['object_id'].str.startswith("TIC")]
#print("KEPLER length: " + str(len(kepler_df)))
#print("Kepler 11449844 " + str(kepler_df.loc[kepler_df['object_id'] == 'KIC 11449844', 'type'].iloc[0]))
#print(kepler_df['type'].value_counts())
#print("TESS length: " + str(len(tess_df)))
#print(tess_df['type'].value_counts())

#iatson.load_model("./IATSON_planet_model_rp95_0.7")
    #iatson.train(training_dir, model_dir, hp) #, continue_from_model='/home/administrator/.watson/0.0.4/IATSON_planet_model_chk_1')
#    iatson.train_cv(training_dir, model_dir, hp, folds=10, plot_metrics=False, retry_indexes=[1, 2, 4, 8], continue_from="./IATSON_planet_model_chk_")
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
#iatson = iatson.load_model("/usr/bin/exoml/IATSON_planet_model/")
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
iatson.predict_test_set_cv("/data/scratch/ml/", "/home/administrator/exoml/08/IATSON_planet_model_chk", batch_size, cores)
#candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
#candidates_df = candidates_df.loc[candidates_df['kepid'] == 11449844]
#iatson.predict_df_cv(training_dir, "/home/users/dfe/mdevora/.watson/0.0.4/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)
