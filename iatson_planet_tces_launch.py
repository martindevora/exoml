import os
import logging
import sys
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog

#from preparation.parser_utils import create_star_csv, CreateStarCsvInput

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import pydevd_pycharm
#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
import shutil
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
import tensorflow_addons as tfa
import pandas as pd
#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from src.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from src.ml.callback.checkpoint import ModelCheckpointCallback
from src.iatson.IATSON_planet import IATSON_planet
from src.ml.learning_rate.schedulers import ExponentialRescaleDecay
from src.ml.loss.cross_entropy import fb_loss, dice_bce_loss, dice_loss
from src.ml.model.base_model import HyperParams

if not isinstance(logging.root, logging.RootLogger):
    logging.root = logging.RootLogger(logging.INFO)
# formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger()
# while len(logger.handlers) > 0:
#     logger.handlers.pop()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
# logger.addHandler(handler)

training_dir = '/data/scratch/ml/'
model_dir = './'
batch_size = 200 #100
lr = 0.00016 #0.00016
warmup_lr = 1e-7
epochs = 400 #400
cores = 10
steps_per_epoch = 148
learning_rate_schedule = ExponentialRescaleDecay(lr,
                                                 decay_steps=20,
                                                 decay_rate=0.98,
                                                 restore_steps=200,
                                                 restore_rate=1.5,
                                                 staircase=True)
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
                 initial_learning_rate=lr, #lr,
                 learning_rate_schedule=learning_rate_schedule,
                 lr_progression=0.9, #0.9
                 gradient_clip_norm=0.01, #0.1,
                 gradient_clip_value=0.05, #0.5,
                 dry_run=False,
                 cores=cores,
                 balance_class_sampling=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 learning_rate_decay=0.98,
                 early_stopping_delta=0,
                 early_stopping_patience=10,
                 class_loss_weights=None, #None,
                 custom_loss=BinaryCrossentropy(),
                 run_eagerly=False,
                 dropout_rate=0.1, # 0.1?
                 dropout_max_rate=0.1, # 0.1?
                 spatial_dropout_rate=0.3, # 0.3?
                 white_noise_std=0.001, #0.001,
                 l1_regularization=0.0005, #0.00005 0.00007-4,8
                 l2_regularization=0.00002, #0.00002,
                 stochastic_weight_average_wait=int(0.75 * epochs),
                 callbacks=[ #ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                            ModelCheckpointCallback(filepath="./IATSON_planet_model_chk",
                                                                original_path='/IATSON_planet_model',
                                                                verbose=0, save_weights_only=False,
                                                                save_best_only=True, monitor='val_r@p99',
                                                                mode='max', save_freq='epoch',
                                                                initial_value_threshold=0.927),
                            WarmUpAndLinDecreaseCallback(lr, lr, warmup_lr, 1, epochs - 1)]
)
shutil.rmtree("./IATSON_planet_model", ignore_errors=True)
iatson = IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                         6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                         10: 'none', 11: 'EB', 12: 'bckEB'},
                        ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                         'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB'],
                        {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'tce_centroids_offset': [0],
                         'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0], 'tce_odd_even': [0], 'none': [0],
                         'EB': [0], 'bckEB': [0], 'candidate': [0], 'tce_candidate': [0]}, hp, mode="all")\
     .build(use_transformers=False, transformer_blocks=1, transformer_heads=2).inform(model_dir)
#iatson.load_model("./IATSON_planet_model_rp95_0.7")
#iatson.train(training_dir, model_dir, hp)
# iatson.train_cv(training_dir, model_dir, hp, folds=10, retry_rp99_threshold=0.9272, retry_indexes=[6],
#                 continue_from="/home/mdevo/exoml/IATSON_planet_model_chk_")

# Current model test metrics
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores)

# Model test metrics with the additional tce_candidates
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           additional_test_set_filename=training_dir + '/q1_q17/injected_objects_tce_candidates.csv')

# Model test metrics with calibration
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='logistic',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='each')
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='platt',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='each')
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='isotonic',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='each')

#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='isotonic',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='global')
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='platt',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='global')
#iatson.predict_test_set_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='logistic',
#                           calibrator_output="/home/mdevo/exoml/IATSON_planet_model",
#                           calibration_mode='global')

# Calibration metrics
#iatson.calibration_metrics_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", batch_size, cores)

# Predict candidates
candidates_df = pd.read_csv(training_dir + '/q1_q17/injected_objects_candidates.csv')
# all_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# # candidates_df = candidates_df.loc[candidates_df['object_id'] == 'KIC 4144236']
#iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict validated candidates only
#candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
#candidates_df = candidates_df.loc[(candidates_df['disc_refname'].str.contains('Valizadegan', na=False)) |
#                                  (candidates_df['disc_refname'].str.contains('Armstrong et al. 2021', na=False))]
#iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict candidates with previously calibrated model
iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores,
                     thresholds=[0.5, 0.99],
                     #validation_df=all_df,
                     from_arrays=True,
                     calibrator_path="/home/mdevo/exoml/IATSON_planet_model_cal_logistic.pkl")

#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
#iatson = iatson.load_model("/usr/bin/exoml/IATSON_planet_model/")
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
