import os
import logging
import numpy as np

from exoml.ml.loss.cross_entropy import binary_focal_loss

#from preparation.parser_utils import create_star_csv, CreateStarCsvInput

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import pydevd_pycharm
#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import shutil
from keras.losses import BinaryCrossentropy
import pandas as pd
#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback
from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.iatson.IATSON_planet import IATSON_planet
from exoml.ml.learning_rate.schedulers import ExponentialRescaleDecay
from exoml.ml.model.base_model import HyperParams

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
model_dir = '/home/administrator/exoml/'
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
                 l2_regularization=0.0005, #0.00002,
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
shutil.rmtree("./option1/IATSON_planet_model", ignore_errors=True)
iatson = (IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                         6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                         10: 'none', 11: 'EB', 12: 'bckEB', 13: 'fp_candidate'},
                        ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                         'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB', 'fp_candidate'],
                        {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'tce_centroids_offset': [0],
                         'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0], 'tce_odd_even': [0], 'none': [0],
                         'EB': [0], 'bckEB': [0], 'candidate': [0], 'tce_candidate': [0], 'fp_candidate': [0]}, hp,
                        mode=['q1q17', 'ete6', 'with_radius', 'no_val', 'with_tce_candidates'])
          .build(use_transformers=False, transformer_blocks=2, transformer_heads=1).inform(model_dir))
#iatson.load_model("./IATSON_planet_model_rp95_0.7")
#iatson.train(training_dir, model_dir, hp)
# iatson.train_cv(training_dir, model_dir, hp, folds=10, retry_rp99_threshold=0.9272, retry_indexes=[6],
#                 continue_from="/home/mdevo/exoml/IATSON_planet_model_chk_")

# Model dataset description
# df = iatson.load_training_set(training_dir=training_dir)
# kepler_df = df.loc[df['object_id'].str.startswith("KIC")]
# tess_df = df.loc[df['object_id'].str.startswith("TIC")]
# print("KEPLER length: " + str(len(kepler_df)))
# print(kepler_df['type'].value_counts())
# print("TESS length: " + str(len(tess_df)))
# print(tess_df['type'].value_counts())

# Current model test metrics
# iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores)

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
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[candidates_df['kepid'] == 11449844]
# # candidates_df = candidates_df.loc[candidates_df['object_id'] == 'KIC 4144236']
#iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict validated candidates only
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[(candidates_df['disc_refname'].str.contains('Valizadegan', na=False)) |
#                                  (candidates_df['disc_refname'].str.contains('Armstrong et al. 2021', na=False))]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict FPs only
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'fp')]
# candidates_df = candidates_df[~candidates_df['radius(earth)'].isna()]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict planet with RV only
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'planet')]
# #candidates_df = candidates_df[~candidates_df['radius(earth)'].isna()]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict planet with RV only or FPs
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'planet') | ((candidates_df['type'] == 'fp') & (~candidates_df['radius(earth)'].isna()))]
# #candidates_df = candidates_df[~candidates_df['radius(earth)'].isna()]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict Santerne et al. 2016 set
candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
candidates_df = candidates_df.loc[
    ((candidates_df['object_id'] == 'KIC 11446443') & (np.abs(candidates_df['period'] - 2.471) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10666592') & (np.abs(candidates_df['period'] - 2.205) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 10748390') & (np.abs(candidates_df['period'] - 4.888) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 6922244') & (np.abs(candidates_df['period'] - 3.522) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 5812701') & (np.abs(candidates_df['period'] - 17.855) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 9941662' ) & (np.abs(candidates_df['period'] - 1.764) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 10874614') & (np.abs(candidates_df['period'] -  3.235) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 8191672') & (np.abs(candidates_df['period'] - 3.548) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 11804465') & (np.abs(candidates_df['period'] - 4.438) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9631995') & (np.abs(candidates_df['period'] - 7.891) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6056992') & (np.abs(candidates_df['period'] - 10.431) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11554435') & (np.abs(candidates_df['period'] - 9.434) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6462863') & (np.abs(candidates_df['period'] - 22.343) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 5780885') & (np.abs(candidates_df['period'] - 4.885) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8359498') & (np.abs(candidates_df['period'] - 3.579) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11359879') & (np.abs(candidates_df['period'] - 4.943) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11974540') & (np.abs(candidates_df['period'] - 24.669) < 0.1))  |
    ((candidates_df['object_id'] == 'KIC 7778437' ) & (np.abs(candidates_df['period'] - 5.014) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9818381' ) & (np.abs(candidates_df['period'] - 3.024) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8506766' ) & (np.abs(candidates_df['period'] - 48.938) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9651668' ) & (np.abs(candidates_df['period'] - 2.684) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11391018' ) & (np.abs(candidates_df['period'] - 30.360) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5771719' ) & (np.abs(candidates_df['period'] - 12.265) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7950644' ) & (np.abs(candidates_df['period'] - 10.291) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9410930' ) & (np.abs(candidates_df['period'] - 1.856) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 2987027' ) & (np.abs(candidates_df['period'] - 17.276) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10666242' ) & (np.abs(candidates_df['period'] - 87.242) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6046540' ) & (np.abs(candidates_df['period'] - 7.341) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6849046' ) & (np.abs(candidates_df['period'] - 4.225) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7877496' ) & (np.abs(candidates_df['period'] - 1.721) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10619192' ) & (np.abs(candidates_df['period'] -  1.486) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9305831' ) & (np.abs(candidates_df['period'] - 3.247) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7046804' ) & (np.abs(candidates_df['period'] - 11.720) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5728139' ) & (np.abs(candidates_df['period'] - 5.334) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10723750' ) & (np.abs(candidates_df['period'] -  50.790) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11046458' ) & (np.abs(candidates_df['period'] - 3.312) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3937519' ) & (np.abs(candidates_df['period'] - 3.413) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10616571' ) & (np.abs(candidates_df['period'] - 23.673) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11442793' ) & (np.abs(candidates_df['period'] -  331.601) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11442793' ) & (np.abs(candidates_df['period'] - 210.603) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6603043' ) & (np.abs(candidates_df['period'] - 110.322) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6471021' ) & (np.abs(candidates_df['period'] - 125.629) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3323887' ) & (np.abs(candidates_df['period'] - 19.271) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3323887' ) & (np.abs(candidates_df['period'] - 38.908) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5449777'  ) & (np.abs(candidates_df['period'] - 7.217) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6289650'  ) & (np.abs(candidates_df['period'] - 166.788) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8219673'  ) & (np.abs(candidates_df['period'] - 20.131) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9478990'  ) & (np.abs(candidates_df['period'] - 21.087) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9967884'  ) & (np.abs(candidates_df['period'] - 5.428) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5779852'  ) & (np.abs(candidates_df['period'] - 252.079) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8890783'  ) & (np.abs(candidates_df['period'] -  58.362) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10395543' ) & (np.abs(candidates_df['period'] - 3.687) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5443837' ) & (np.abs(candidates_df['period'] - 3.658) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5441980'  ) & (np.abs(candidates_df['period'] - 5.894) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6309763'  ) & (np.abs(candidates_df['period'] - 3.252) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7368664'  ) & (np.abs(candidates_df['period'] - 12.875) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9846086'  ) & (np.abs(candidates_df['period'] - 37.865) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11773022'  ) & (np.abs(candidates_df['period'] - 45.155) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11773022'  ) & (np.abs(candidates_df['period'] - 130.177) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7529266'  ) & (np.abs(candidates_df['period'] - 8.600) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7906882'  ) & (np.abs(candidates_df['period'] - 52.514) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8891278'  ) & (np.abs(candidates_df['period'] - 12.719) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4044005'  ) & (np.abs(candidates_df['period'] - 17.512) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3441784'  ) & (np.abs(candidates_df['period'] - 52.569) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 2309719'  ) & (np.abs(candidates_df['period'] - 54.356) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3247268'  ) & (np.abs(candidates_df['period'] - 86.679) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6629332'  ) & (np.abs(candidates_df['period'] - 2.155) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6470149'  ) & (np.abs(candidates_df['period'] - 165.742) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6665223'  ) & (np.abs(candidates_df['period'] - 119.408) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8751933'  ) & (np.abs(candidates_df['period'] - 86.648) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8631160'  ) & (np.abs(candidates_df['period'] - 162.054) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4639868'  ) & (np.abs(candidates_df['period'] - 53.101) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7303287'  ) & (np.abs(candidates_df['period'] - 125.865) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8958035'  ) & (np.abs(candidates_df['period'] - 7.981) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9425139'  ) & (np.abs(candidates_df['period'] - 305.076) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11122894'  ) & (np.abs(candidates_df['period'] - 74.928) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11122894'  ) & (np.abs(candidates_df['period'] - 150.019) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11075279'  ) & (np.abs(candidates_df['period'] - 345.160) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7449844' ) & (np.abs(candidates_df['period'] - 1.152) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11702948' ) & (np.abs(candidates_df['period'] - 9.771) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 12365184' ) & (np.abs(candidates_df['period'] - 69.727) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11909686' ) & (np.abs(candidates_df['period'] - 185.953) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5475431' ) & (np.abs(candidates_df['period'] - 0.918) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10028792' ) & (np.abs(candidates_df['period'] - 114.736) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11045383' ) & (np.abs(candidates_df['period'] - 41.167) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4570949' ) & (np.abs(candidates_df['period'] - 1.545) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10005758' ) & (np.abs(candidates_df['period'] - 134.479) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10158418' ) & (np.abs(candidates_df['period'] - 5.007) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 3128793' ) & (np.abs(candidates_df['period'] - 124.686) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 2975770' ) & (np.abs(candidates_df['period'] - 71.525) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6716021' ) & (np.abs(candidates_df['period'] - 110.756) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6025124' ) & (np.abs(candidates_df['period'] - 26.838) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 12735740' ) & (np.abs(candidates_df['period'] - 282.525) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4150804' ) & (np.abs(candidates_df['period'] - 160.885) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9025971' ) & (np.abs(candidates_df['period'] - 141.242) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 2581316' ) & (np.abs(candidates_df['period'] - 217.832) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10795103' ) & (np.abs(candidates_df['period'] - 214.311) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8494410'  ) & (np.abs(candidates_df['period'] - 208.869) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7017372'  ) & (np.abs(candidates_df['period'] - 5.241) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 10735331'  ) & (np.abs(candidates_df['period'] - 213.399) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7763269'  ) & (np.abs(candidates_df['period'] - 6.408) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6775985'  ) & (np.abs(candidates_df['period'] - 27.961) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11357192'  ) & (np.abs(candidates_df['period'] - 186.517) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9533489'  ) & (np.abs(candidates_df['period'] - 197.146) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5688997' ) & (np.abs(candidates_df['period'] - 23.871) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7813039'  ) & (np.abs(candidates_df['period'] - 141.734) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4638237'  ) & (np.abs(candidates_df['period'] - 290.140) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4058169'  ) & (np.abs(candidates_df['period'] - 282.536) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4769799'  ) & (np.abs(candidates_df['period'] - 21.929) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5179609'  ) & (np.abs(candidates_df['period'] - 43.931) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7377343'  ) & (np.abs(candidates_df['period'] - 7.954) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7837526'  ) & (np.abs(candidates_df['period'] - 28.297) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8509781'  ) & (np.abs(candidates_df['period'] - 70.336) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9724993'  ) & (np.abs(candidates_df['period'] - 7.863) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9962595'  ) & (np.abs(candidates_df['period'] - 11.374) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 12645761'  ) & (np.abs(candidates_df['period'] - 2.710) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 9221398'  ) & (np.abs(candidates_df['period'] - 13.788) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5629353'  ) & (np.abs(candidates_df['period'] - 33.320) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8197761'  ) & (np.abs(candidates_df['period'] - 9.869) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 11147460'  ) & (np.abs(candidates_df['period'] - 2.054) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 4851217'  ) & (np.abs(candidates_df['period'] - 1.235) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 5598639'  ) & (np.abs(candidates_df['period'] - 0.649) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 6965293'  ) & (np.abs(candidates_df['period'] - 2.539) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7431665'  ) & (np.abs(candidates_df['period'] - 281.544) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 7938468'  ) & (np.abs(candidates_df['period'] - 7.227) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8453324'  ) & (np.abs(candidates_df['period'] - 1.262) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8552540'  ) & (np.abs(candidates_df['period'] - 0.531) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 8590527'  ) & (np.abs(candidates_df['period'] - 0.740) < 0.1)) |
    ((candidates_df['object_id'] == 'KIC 12365000' ) & (np.abs(candidates_df['period'] - 1.263) < 0.1))
]
candidates_df = candidates_df.loc[candidates_df['close_period'] == 0]
iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
                    thresholds=[0.5],
                    #validation_df=all_df,
                    from_arrays=True)

# Predict ETE6 planets only
# candidates_df = pd.read_csv(training_dir + '/ete6/injected_objects.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'planet')]
# candidates_df = candidates_df[~candidates_df['radius(earth)'].isna()]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict ETE6 FPs only
# candidates_df = pd.read_csv(training_dir + '/ete6/injected_objects.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'EB') | (candidates_df['type'] == 'bckEB')]
# candidates_df = candidates_df[~candidates_df['radius(earth)'].isna()]
# iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# Predict candidates with previously calibrated model
# iatson.predict_df_cv(training_dir, "/home/martin/.watson/0.0.4/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                      thresholds=[0.5, 0.99],
#                      #validation_df=all_df,
#                      from_arrays=False)
#                      #calibrator_path="/home/mdevo/exoml/IATSON_planet_model_cal_logistic.pkl")
# iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores, from_arrays=False)

#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
#iatson = iatson.load_model("/usr/bin/exoml/IATSON_planet_model/")
#iatson.predict_test_set(training_dir, "./IATSON_planet_model", batch_size, cores)
