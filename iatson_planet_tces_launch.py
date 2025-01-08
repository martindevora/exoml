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
                        mode=['tess', 'with_radius', 'no_val', 'with_random_folded', 'with_fp'])
          .build(use_transformers=False, transformer_blocks=2, transformer_heads=1).inform(model_dir))
#iatson.load_model("./IATSON_planet_model_rp95_0.7")
#iatson.train(training_dir, model_dir, hp)
# iatson.fine_tune_cv(training_dir, model_dir, hp, regenerate_fine_tuning_set=False,
#                     continue_from="/home/administrator/exoml/IATSON_planet_model_chk_", folds=10,
#                     retry_indexes=[0])


# MODEL DATASET DESCRIPTION
# df = iatson.load_training_set(training_dir=training_dir)
# kepler_df = df.loc[df['object_id'].str.startswith("KIC")]
# tess_df = df.loc[df['object_id'].str.startswith("TIC")]
# print("KEPLER length: " + str(len(kepler_df)))
# print(kepler_df['type'].value_counts())
# print("TESS length: " + str(len(tess_df)))
# print(tess_df['type'].value_counts())


# ORIGINAL MODEL TEST METRICS
# iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores)

# CALIBRATE MODELS
# iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='platt')
# iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores,
#                           calibration_method='isotonic')


# MODEL TEST METRICS WITH CALIBRATION
# lp_threshold = 0.806
# vp_threshold = 0.968
# ln_threshold = 0.042
# vn_threshold = 0.001
# results = iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores,
#                           calibrator_path="IATSON_planet_cal_platt.pkl")
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"predictions_map_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
# tps_above_lp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > lp_threshold))]
# tps_above_vp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > vp_threshold))]
# tps_above_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] >= 0.5))]
# fps_above_lp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold))]
# fps_above_vp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > vp_threshold))]
# fps_above_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] >= 0.5))]
# print(f"TPs above lp: {tps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_lp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above vp: {tps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_vp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above 0.5: {tps_above_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FPs above lp: {fps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_lp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above vp: {fps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_vp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above 0.5: {fps_above_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# tns_below_ln = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < ln_threshold))]
# tns_below_vn = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < vn_threshold))]
# tns_below_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] <= 0.5))]
# fns_below_ln = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold))]
# fns_below_vn = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < vn_threshold))]
# fns_below_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] <= 0.5))]
# print(f"TNs below ln: {tns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_ln.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below vn: {tns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_vn.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below 0.5: {tns_below_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FNs below ln: {fns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_ln.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below vn: {fns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_vn.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below 0.5: {fns_below_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# results = results.loc[(((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold)) |
#                       ((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold)))]
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"fps_fns_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
lp_threshold = 0.646
vp_threshold = 0.961
ln_threshold = 0.056
vn_threshold = 0.007
# results = iatson.predict_test_set_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores,
#                           calibrator_path="IATSON_planet_cal_isotonic.pkl")
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"predictions_map_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
# tps_above_lp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > lp_threshold))]
# tps_above_vp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > vp_threshold))]
# tps_above_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] >= 0.5))]
# fps_above_lp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold))]
# fps_above_vp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > vp_threshold))]
# fps_above_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] >= 0.5))]
# print(f"TPs above lp: {tps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_lp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above vp: {tps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_vp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above 0.5: {tps_above_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FPs above lp: {fps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_lp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above vp: {fps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_vp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above 0.5: {fps_above_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# tns_below_ln = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < ln_threshold))]
# tns_below_vn = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < vn_threshold))]
# tns_below_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] <= 0.5))]
# fns_below_ln = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold))]
# fns_below_vn = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < vn_threshold))]
# fns_below_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] <= 0.5))]
# print(f"TNs below ln: {tns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_ln.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below vn: {tns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_vn.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below 0.5: {tns_below_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FNs below ln: {fns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_ln.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below vn: {fns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_vn.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below 0.5: {fns_below_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# results = results.loc[(((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold)) |
#                       ((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold)))]
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"fps_fns_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )

# CALIBRATION METRICS
#iatson.calibration_metrics_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", batch_size, cores)

# Predict candidates
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[candidates_df['kepid'] == 11449844]
# # candidates_df = candidates_df.loc[candidates_df['object_id'] == 'KIC 4144236']
#iatson.predict_df_cv(training_dir, "/home/mdevo/exoml/IATSON_planet_model_chk", candidates_df, batch_size, cores,
#                     thresholds=[0.5, 0.99],
#                     #validation_df=all_df,
#                     from_arrays=True)

# PREDICT VALIDATED PLANETS
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_armstrong_df = candidates_df.loc[(candidates_df['disc_refname'].str.contains('Armstrong et al. 2021', na=False))]
# results = iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_armstrong_df, batch_size, cores,
#                     thresholds=[0.5], calibrator_path="IATSON_planet_cal_isotonic.pkl",
#                     #validation_df=all_df,
#                     from_arrays=True)
# results = results[0.5]
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"predictions_map_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
# tps_above_lp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > lp_threshold))]
# tps_above_vp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > vp_threshold))]
# tps_above_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] >= 0.5))]
# fps_above_lp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold))]
# fps_above_vp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > vp_threshold))]
# fps_above_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] >= 0.5))]
# print(f"TPs above lp: {tps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_lp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above vp: {tps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_vp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above 0.5: {tps_above_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# candidates_exominer_df = candidates_df.loc[(candidates_df['disc_refname'].str.contains('Valizadegan', na=False))]
# candidates_exominer_df = candidates_exominer_df.loc[~candidates_exominer_df['object_id'].str.contains("7677005")]
# results = iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", candidates_exominer_df, batch_size, cores,
#                     thresholds=[0.5], calibrator_path="IATSON_planet_cal_isotonic.pkl",
#                     #validation_df=all_df,
#                     from_arrays=True)
# results = results[0.5]
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"predictions_map_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
# tps_above_lp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > lp_threshold))]
# tps_above_vp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > vp_threshold))]
# tps_above_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] >= 0.5))]
# fps_above_lp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold))]
# fps_above_vp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > vp_threshold))]
# fps_above_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] >= 0.5))]
# print(f"TPs above lp: {tps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_lp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above vp: {tps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_vp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above 0.5: {tps_above_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")


# Predict FPs only
# candidates_df = pd.read_csv(training_dir + '/q1_q17/classified_tces.csv')
# candidates_df = candidates_df.loc[(candidates_df['type'] == 'fp')]
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

# Predict Santerne labels Santerne et al. 2016 https://www.aanda.org/articles/aa/pdf/2016/03/aa27329-15.pdf
# candidates_santerne_df = iatson.predict_test_set_cv(training_dir, f"{model_dir}/08/IATSON_planet_model_chk", batch_size, cores,
#                             calibrator_path="IATSON_planet_cal_isotonic.pkl")
# candidates_santerne_df = candidates_santerne_df.loc[
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11446443')) & (np.abs(candidates_santerne_df['period'] - 2.471) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10666592')) & (np.abs(candidates_santerne_df['period'] - 2.205) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10748390')) & (np.abs(candidates_santerne_df['period'] - 4.888) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6922244')) & (np.abs(candidates_santerne_df['period'] - 3.522) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5812701')) & (np.abs(candidates_santerne_df['period'] - 17.855) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9941662')) & (np.abs(candidates_santerne_df['period'] - 1.764) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10874614')) & (np.abs(candidates_santerne_df['period'] -  3.235) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8191672')) & (np.abs(candidates_santerne_df['period'] - 3.548) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11804465')) & (np.abs(candidates_santerne_df['period'] - 4.438) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9631995')) & (np.abs(candidates_santerne_df['period'] - 7.891) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6056992')) & (np.abs(candidates_santerne_df['period'] - 10.431) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11554435')) & (np.abs(candidates_santerne_df['period'] - 9.434) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6462863')) & (np.abs(candidates_santerne_df['period'] - 22.343) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5780885')) & (np.abs(candidates_santerne_df['period'] - 4.885) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8359498')) & (np.abs(candidates_santerne_df['period'] - 3.579) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11359879')) & (np.abs(candidates_santerne_df['period'] - 4.943) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11974540')) & (np.abs(candidates_santerne_df['period'] - 24.669) < 0.1))  |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7778437')) & (np.abs(candidates_santerne_df['period'] - 5.014) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9818381')) & (np.abs(candidates_santerne_df['period'] - 3.024) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8506766')) & (np.abs(candidates_santerne_df['period'] - 48.938) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9651668')) & (np.abs(candidates_santerne_df['period'] - 2.684) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11391018')) & (np.abs(candidates_santerne_df['period'] - 30.360) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5771719')) & (np.abs(candidates_santerne_df['period'] - 12.265) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7950644')) & (np.abs(candidates_santerne_df['period'] - 10.291) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9410930')) & (np.abs(candidates_santerne_df['period'] - 1.856) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 2987027')) & (np.abs(candidates_santerne_df['period'] - 17.276) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10666242')) & (np.abs(candidates_santerne_df['period'] - 87.242) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6046540')) & (np.abs(candidates_santerne_df['period'] - 7.341) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6849046')) & (np.abs(candidates_santerne_df['period'] - 4.225) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7877496')) & (np.abs(candidates_santerne_df['period'] - 1.721) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10619192')) & (np.abs(candidates_santerne_df['period'] -  1.486) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9305831')) & (np.abs(candidates_santerne_df['period'] - 3.247) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7046804')) & (np.abs(candidates_santerne_df['period'] - 11.720) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5728139')) & (np.abs(candidates_santerne_df['period'] - 5.334) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10723750')) & (np.abs(candidates_santerne_df['period'] -  50.790) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11046458')) & (np.abs(candidates_santerne_df['period'] - 3.312) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3937519')) & (np.abs(candidates_santerne_df['period'] - 3.413) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10616571')) & (np.abs(candidates_santerne_df['period'] - 23.673) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11442793')) & (np.abs(candidates_santerne_df['period'] -  331.601) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11442793')) & (np.abs(candidates_santerne_df['period'] - 210.603) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6603043')) & (np.abs(candidates_santerne_df['period'] - 110.322) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6471021')) & (np.abs(candidates_santerne_df['period'] - 125.629) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3323887')) & (np.abs(candidates_santerne_df['period'] - 19.271) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3323887')) & (np.abs(candidates_santerne_df['period'] - 38.908) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5449777')) & (np.abs(candidates_santerne_df['period'] - 7.217) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6289650')) & (np.abs(candidates_santerne_df['period'] - 166.788) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8219673')) & (np.abs(candidates_santerne_df['period'] - 20.131) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9478990')) & (np.abs(candidates_santerne_df['period'] - 21.087) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9967884')) & (np.abs(candidates_santerne_df['period'] - 5.428) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5779852')) & (np.abs(candidates_santerne_df['period'] - 252.079) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8890783')) & (np.abs(candidates_santerne_df['period'] -  58.362) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10395543')) & (np.abs(candidates_santerne_df['period'] - 3.687) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5443837')) & (np.abs(candidates_santerne_df['period'] - 3.658) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5441980')) & (np.abs(candidates_santerne_df['period'] - 5.894) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6309763')) & (np.abs(candidates_santerne_df['period'] - 3.252) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7368664')) & (np.abs(candidates_santerne_df['period'] - 12.875) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9846086')) & (np.abs(candidates_santerne_df['period'] - 37.865) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11773022')) & (np.abs(candidates_santerne_df['period'] - 45.155) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11773022')) & (np.abs(candidates_santerne_df['period'] - 130.177) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7529266')) & (np.abs(candidates_santerne_df['period'] - 8.600) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7906882')) & (np.abs(candidates_santerne_df['period'] - 52.514) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8891278')) & (np.abs(candidates_santerne_df['period'] - 12.719) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4044005')) & (np.abs(candidates_santerne_df['period'] - 17.512) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3441784')) & (np.abs(candidates_santerne_df['period'] - 52.569) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 2309719')) & (np.abs(candidates_santerne_df['period'] - 54.356) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3247268')) & (np.abs(candidates_santerne_df['period'] - 86.679) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6629332')) & (np.abs(candidates_santerne_df['period'] - 2.155) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6470149')) & (np.abs(candidates_santerne_df['period'] - 165.742) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6665223')) & (np.abs(candidates_santerne_df['period'] - 119.408) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8751933')) & (np.abs(candidates_santerne_df['period'] - 86.648) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8631160')) & (np.abs(candidates_santerne_df['period'] - 162.054) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4639868')) & (np.abs(candidates_santerne_df['period'] - 53.101) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7303287')) & (np.abs(candidates_santerne_df['period'] - 125.865) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8958035')) & (np.abs(candidates_santerne_df['period'] - 7.981) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9425139')) & (np.abs(candidates_santerne_df['period'] - 305.076) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11122894')) & (np.abs(candidates_santerne_df['period'] - 74.928) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11122894')) & (np.abs(candidates_santerne_df['period'] - 150.019) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11075279')) & (np.abs(candidates_santerne_df['period'] - 345.160) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7449844')) & (np.abs(candidates_santerne_df['period'] - 1.152) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11702948')) & (np.abs(candidates_santerne_df['period'] - 9.771) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 12365184')) & (np.abs(candidates_santerne_df['period'] - 69.727) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11909686')) & (np.abs(candidates_santerne_df['period'] - 185.953) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5475431')) & (np.abs(candidates_santerne_df['period'] - 0.918) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10028792')) & (np.abs(candidates_santerne_df['period'] - 114.736) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11045383')) & (np.abs(candidates_santerne_df['period'] - 41.167) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4570949')) & (np.abs(candidates_santerne_df['period'] - 1.545) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10005758')) & (np.abs(candidates_santerne_df['period'] - 134.479) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10158418')) & (np.abs(candidates_santerne_df['period'] - 5.007) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 3128793')) & (np.abs(candidates_santerne_df['period'] - 124.686) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 2975770')) & (np.abs(candidates_santerne_df['period'] - 71.525) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6716021')) & (np.abs(candidates_santerne_df['period'] - 110.756) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6025124')) & (np.abs(candidates_santerne_df['period'] - 26.838) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 12735740')) & (np.abs(candidates_santerne_df['period'] - 282.525) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4150804')) & (np.abs(candidates_santerne_df['period'] - 160.885) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9025971')) & (np.abs(candidates_santerne_df['period'] - 141.242) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 2581316')) & (np.abs(candidates_santerne_df['period'] - 217.832) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10795103')) & (np.abs(candidates_santerne_df['period'] - 214.311) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8494410')) & (np.abs(candidates_santerne_df['period'] - 208.869) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7017372')) & (np.abs(candidates_santerne_df['period'] - 5.241) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 10735331')) & (np.abs(candidates_santerne_df['period'] - 213.399) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7763269')) & (np.abs(candidates_santerne_df['period'] - 6.408) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6775985')) & (np.abs(candidates_santerne_df['period'] - 27.961) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11357192')) & (np.abs(candidates_santerne_df['period'] - 186.517) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9533489')) & (np.abs(candidates_santerne_df['period'] - 197.146) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5688997')) & (np.abs(candidates_santerne_df['period'] - 23.871) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7813039')) & (np.abs(candidates_santerne_df['period'] - 141.734) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4638237')) & (np.abs(candidates_santerne_df['period'] - 290.140) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4058169')) & (np.abs(candidates_santerne_df['period'] - 282.536) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4769799')) & (np.abs(candidates_santerne_df['period'] - 21.929) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5179609')) & (np.abs(candidates_santerne_df['period'] - 43.931) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7377343')) & (np.abs(candidates_santerne_df['period'] - 7.954) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7837526')) & (np.abs(candidates_santerne_df['period'] - 28.297) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8509781')) & (np.abs(candidates_santerne_df['period'] - 70.336) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9724993')) & (np.abs(candidates_santerne_df['period'] - 7.863) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9962595')) & (np.abs(candidates_santerne_df['period'] - 11.374) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 12645761')) & (np.abs(candidates_santerne_df['period'] - 2.710) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 9221398')) & (np.abs(candidates_santerne_df['period'] - 13.788) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5629353')) & (np.abs(candidates_santerne_df['period'] - 33.320) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8197761')) & (np.abs(candidates_santerne_df['period'] - 9.869) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 11147460')) & (np.abs(candidates_santerne_df['period'] - 2.054) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 4851217')) & (np.abs(candidates_santerne_df['period'] - 1.235) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 5598639')) & (np.abs(candidates_santerne_df['period'] - 0.649) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 6965293')) & (np.abs(candidates_santerne_df['period'] - 2.539) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7431665')) & (np.abs(candidates_santerne_df['period'] - 281.544) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 7938468')) & (np.abs(candidates_santerne_df['period'] - 7.227) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8453324')) & (np.abs(candidates_santerne_df['period'] - 1.262) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8552540')) & (np.abs(candidates_santerne_df['period'] - 0.531) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 8590527')) & (np.abs(candidates_santerne_df['period'] - 0.740) < 0.1)) |
#  ((candidates_santerne_df['object_id'].str.startswith('KIC 12365000')) & (np.abs(candidates_santerne_df['period'] - 1.263) < 0.1))
# ]
# candidates_santerne_df['santerne_label'] = 'EB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7778437')) &
#                            ((np.abs(candidates_santerne_df['period'] - 5.014) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 2987027')) &
#                            ((np.abs(candidates_santerne_df['period'] - 17.276) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 3937519')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.413) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6471021')) &
#                            ((np.abs(candidates_santerne_df['period'] - 125.629) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5449777')) &
#                            ((np.abs(candidates_santerne_df['period'] - 7.217) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8890783')) &
#                            ((np.abs(candidates_santerne_df['period'] - 58.362) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6309763')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.252) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8631160')) &
#                            ((np.abs(candidates_santerne_df['period'] - 162.054) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9425139')) &
#                            ((np.abs(candidates_santerne_df['period'] - 305.076) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11122894')) &
#                            ((np.abs(candidates_santerne_df['period'] - 150.019) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11075279')) &
#                            ((np.abs(candidates_santerne_df['period'] - 345.160) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10005758')) &
#                            ((np.abs(candidates_santerne_df['period'] - 134.479) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 2975770')) &
#                            ((np.abs(candidates_santerne_df['period'] - 71.525) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6716021')) &
#                            ((np.abs(candidates_santerne_df['period'] - 110.756) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 4150804')) &
#                            ((np.abs(candidates_santerne_df['period'] - 160.885) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10795103')) &
#                            ((np.abs(candidates_santerne_df['period'] - 214.311) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7017372')) &
#                            ((np.abs(candidates_santerne_df['period'] - 5.241) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5629353')) &
#                            ((np.abs(candidates_santerne_df['period'] -  33.320) < 0.1)), ['santerne_label']] = 'candidate'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11446443')) &
#                            ((np.abs(candidates_santerne_df['period'] - 2.471) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10666592')) &
#                            ((np.abs(candidates_santerne_df['period'] - 2.205) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10748390')) &
#                            ((np.abs(candidates_santerne_df['period'] - 4.888) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6922244')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.522) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5812701')) &
#                            ((np.abs(candidates_santerne_df['period'] - 17.855) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9941662')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.764) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10874614')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.235) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8191672')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.548) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11804465')) &
#                            ((np.abs(candidates_santerne_df['period'] - 4.438) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9631995')) &
#                            ((np.abs(candidates_santerne_df['period'] - 7.891) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6056992')) &
#                            ((np.abs(candidates_santerne_df['period'] - 10.431) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11554435')) &
#                            ((np.abs(candidates_santerne_df['period'] - 9.434) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6462863')) &
#                            ((np.abs(candidates_santerne_df['period'] - 22.343) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5780885')) &
#                            ((np.abs(candidates_santerne_df['period'] - 4.885) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8359498')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.579) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11359879')) &
#                            ((np.abs(candidates_santerne_df['period'] - 4.943) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9818381')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.024) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9651668')) &
#                            ((np.abs(candidates_santerne_df['period'] - 2.684) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5771719')) &
#                            ((np.abs(candidates_santerne_df['period'] - 12.265) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7950644')) &
#                            ((np.abs(candidates_santerne_df['period'] - 10.291) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9410930')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.856) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6046540')) &
#                            ((np.abs(candidates_santerne_df['period'] - 7.341) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6849046')) &
#                            ((np.abs(candidates_santerne_df['period'] - 4.225) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7877496')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.721) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10619192')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.486) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9305831')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.247) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7046804')) &
#                            ((np.abs(candidates_santerne_df['period'] - 11.720) < 0.1)), ['santerne_label']] = 'BD'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5728139')) &
#                            ((np.abs(candidates_santerne_df['period'] - 5.334) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10723750')) &
#                            ((np.abs(candidates_santerne_df['period'] - 50.790) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11046458')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.312) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11442793')) &
#                            ((np.abs(candidates_santerne_df['period'] - 331.601) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11442793')) &
#                            ((np.abs(candidates_santerne_df['period'] - 210.603) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 3323887')) &
#                            ((np.abs(candidates_santerne_df['period'] - 19.271) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 3323887')) &
#                            ((np.abs(candidates_santerne_df['period'] - 38.908) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 6289650')) &
#                            ((np.abs(candidates_santerne_df['period'] - 166.788) < 0.1)), ['santerne_label']] = 'BD'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9478990')) &
#                            ((np.abs(candidates_santerne_df['period'] - 21.087) < 0.1)), ['santerne_label']] = 'BD'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5779852')) &
#                            ((np.abs(candidates_santerne_df['period'] - 252.079) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10395543')) &
#                            ((np.abs(candidates_santerne_df['period'] - 3.687) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7368664')) &
#                            ((np.abs(candidates_santerne_df['period'] - 12.875) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11773022')) &
#                            ((np.abs(candidates_santerne_df['period'] -  45.155) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11773022')) &
#                            ((np.abs(candidates_santerne_df['period'] - 130.177) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7529266')) &
#                            ((np.abs(candidates_santerne_df['period'] - 8.600) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 3441784')) &
#                            ((np.abs(candidates_santerne_df['period'] - 52.569) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 3247268')) &
#                            ((np.abs(candidates_santerne_df['period'] - 86.679) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8751933')) &
#                            ((np.abs(candidates_santerne_df['period'] - 86.648) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7303287')) &
#                            ((np.abs(candidates_santerne_df['period'] - 125.865) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11122894')) &
#                            ((np.abs(candidates_santerne_df['period'] - 74.928) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 12365184')) &
#                            ((np.abs(candidates_santerne_df['period'] - 69.727) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 5475431')) &
#                            ((np.abs(candidates_santerne_df['period'] - 0.918) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10028792')) &
#                            ((np.abs(candidates_santerne_df['period'] - 114.736) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 4570949')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.545) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 10158418')) &
#                            ((np.abs(candidates_santerne_df['period'] - 5.007) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 12735740')) &
#                            ((np.abs(candidates_santerne_df['period'] - 282.525) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9025971')) &
#                            ((np.abs(candidates_santerne_df['period'] - 141.242) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 2581316')) &
#                            ((np.abs(candidates_santerne_df['period'] - 217.832) < 0.1)), ['santerne_label']] = 'planet'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8494410')) &
#                            ((np.abs(candidates_santerne_df['period'] - 208.869) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7763269')) &
#                            ((np.abs(candidates_santerne_df['period'] -  6.408) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 9533489')) &
#                            ((np.abs(candidates_santerne_df['period'] - 197.146) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 7377343')) &
#                            ((np.abs(candidates_santerne_df['period'] - 7.954) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8509781')) &
#                            ((np.abs(candidates_santerne_df['period'] - 70.336) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 12645761')) &
#                            ((np.abs(candidates_santerne_df['period'] - 2.710) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 11147460')) &
#                            ((np.abs(candidates_santerne_df['period'] - 2.054) < 0.1)), ['santerne_label']] = 'CEB'
# candidates_santerne_df.loc[(candidates_santerne_df['object_id'].str.startswith('KIC 8453324')) &
#                            ((np.abs(candidates_santerne_df['period'] - 1.262) < 0.1)), ['santerne_label']] = 'CEB'
# #candidates_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['close_period'] == 0]
# planets_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['santerne_label'] == 'planet']
# ebs_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['santerne_label'] == 'EB']
# cebs_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['santerne_label'] == 'CEB']
# bds_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['santerne_label'] == 'BD']
# uncertain_santerne_df = candidates_santerne_df.loc[candidates_santerne_df['santerne_label'] == 'candidate']
# logging.info(f"Predictions Santerne: \n" + candidates_santerne_df.loc[candidates_santerne_df['expected_class'] == 1].to_string())
# tps = []
# tns = []
# fps = []
# fns = []
# for threshold in [vn_threshold, ln_threshold, 0.5, lp_threshold, vp_threshold]:
#     threshold_tps = candidates_santerne_df.loc[
#         (candidates_santerne_df['expected_class'] == 1) & (candidates_santerne_df['prediction_value'] >= threshold)].shape[0]
#     threshold_fps = candidates_santerne_df.loc[
#         (candidates_santerne_df['expected_class'] == 0) & (candidates_santerne_df['prediction_value'] >= threshold)].shape[0]
#     threshold_tns = candidates_santerne_df.loc[
#         (candidates_santerne_df['expected_class'] == 0) & (candidates_santerne_df['prediction_value'] < threshold)].shape[0]
#     threshold_fns = candidates_santerne_df.loc[
#         (candidates_santerne_df['expected_class'] == 1) & (candidates_santerne_df['prediction_value'] < threshold)].shape[0]
#     tps = tps + [threshold_tps]
#     tns = tns + [threshold_tns]
#     fps = fps + [threshold_fps]
#     fns = fns + [threshold_fns]
#     logging.info(f"Santerne threshold={threshold} tps={threshold_tps}, tns={threshold_tns}, fps={threshold_fps}, fns={threshold_fns}")
#     logging.info(f"Santerne threshold={threshold} accuracy={(threshold_tps + threshold_tns) / (threshold_tps + threshold_tns + threshold_fps + threshold_fns)}" )
# logging.info(f"Predictions Santerne planets: \n" + planets_santerne_df.to_string())
# logging.info(f"Santerne planets Mean prediction value={np.nanmean(planets_santerne_df['prediction_value'])}+-{np.nanstd(planets_santerne_df['prediction_value'])}")
# logging.info(f"Santerne planets Median prediction value={np.nanmedian(planets_santerne_df['prediction_value'])}+-{np.nanstd(planets_santerne_df['prediction_value'])}")
# logging.info(f"Predictions Santerne EBs: \n" + ebs_santerne_df.to_string())
# logging.info(f"Santerne EBs Mean prediction value={np.nanmean(ebs_santerne_df['prediction_value'])}+-{np.nanstd(ebs_santerne_df['prediction_value'])}")
# logging.info(f"Santerne EBs Median prediction value={np.nanmedian(ebs_santerne_df['prediction_value'])}+-{np.nanstd(ebs_santerne_df['prediction_value'])}")
# logging.info(f"Predictions Santerne CEBs: \n" + cebs_santerne_df.to_string())
# logging.info(f"Santerne CEBs Mean prediction value={np.nanmean(cebs_santerne_df['prediction_value'])}+-{np.nanstd(cebs_santerne_df['prediction_value'])}")
# logging.info(f"Santerne CEBs Median prediction value={np.nanmedian(cebs_santerne_df['prediction_value'])}+-{np.nanstd(cebs_santerne_df['prediction_value'])}")
# logging.info(f"Predictions Santerne BDs: \n" + bds_santerne_df.to_string())
# logging.info(f"Santerne BDs Mean prediction value={np.nanmean(bds_santerne_df['prediction_value'])}+-{np.nanstd(bds_santerne_df['prediction_value'])}")
# logging.info(f"Santerne BDs Median prediction value={np.nanmedian(bds_santerne_df['prediction_value'])}+-{np.nanstd(bds_santerne_df['prediction_value'])}")
# logging.info(f"Predictions Santerne uncertain: \n" + uncertain_santerne_df.to_string())

# Radial velocities + negatives dataset predictions
# rv_planets_df = iatson.predict_test_set_cv(training_dir, f"{model_dir}/08/IATSON_planet_model_chk", batch_size, cores,
#                             calibrator_path="IATSON_planet_cal_isotonic.pkl")
# rv_planets_df = rv_planets_df.loc[rv_planets_df['type'] != 'planet_transit']
# only_rv_planets_df = rv_planets_df.loc[rv_planets_df['type'] == 'planet']
# logging.info(f"Predictions RV planets: \n" + only_rv_planets_df.to_string())
# tps = []
# tns = []
# fps = []
# fns = []
# for threshold in [vn_threshold, ln_threshold, 0.5, lp_threshold, vp_threshold]:
#     threshold_tps = rv_planets_df.loc[
#         (rv_planets_df['expected_class'] == 1) & (rv_planets_df['prediction_value'] > threshold)].shape[0]
#     threshold_fps = rv_planets_df.loc[
#         (rv_planets_df['expected_class'] == 0) & (rv_planets_df['prediction_value'] > threshold)].shape[0]
#     threshold_tns = rv_planets_df.loc[
#         (rv_planets_df['expected_class'] == 0) & (rv_planets_df['prediction_value'] <= threshold)].shape[0]
#     threshold_fns = rv_planets_df.loc[
#         (rv_planets_df['expected_class'] == 1) & (rv_planets_df['prediction_value'] <= threshold)].shape[0]
#     tps = tps + [threshold_tps]
#     tns = tns + [threshold_tns]
#     fps = fps + [threshold_fps]
#     fns = fns + [threshold_fns]
#     logging.info(f"RV planets + negatives threshold={threshold} tps={threshold_tps}, tns={threshold_tns}, fps={threshold_fps}, fns={threshold_fns}")
#     logging.info(f"RV planets + negatives threshold={threshold} "
#                  f"accuracy={(threshold_tps + threshold_tns) / (threshold_tps + threshold_tns + threshold_fps + threshold_fns)} "
#                  f"precision={(threshold_tps) / (threshold_tps + threshold_fps)} "
#                  f"recall={(threshold_tps) / (threshold_tps + threshold_fns)}" )


# PREDICT TESS TEST SET
tess_df = pd.read_csv("/data/scratch/ml/tess/tess_tois.csv")
tess_df = tess_df.loc[~tess_df['object_id'].isin(["TIC 9837861",
"TIC 114103136",
"TIC 303867090",
"TIC 365683032",
"TIC 233059608",
"TIC 349790953",
"TIC 241102583",
"TIC 124709665",
"TIC 374847873",
"TIC 244089109",
"TIC 269450900",
"TIC 317520667",
"TIC 233602827",
"TIC 367366318",
"TIC 242460075",
"TIC 98286422",
"TIC 110178537",
"TIC 305512837",
"TIC 20892672",
"TIC 167227214",
"TIC 194179013",
"TIC 16102323",
"TIC 350153977",
"TIC 23863106",
"TIC 325670253",
"TIC 183979262",
"TIC 331034236",
"TIC 387527558",
"TIC 257060897",
"TIC 284326455",
"TIC 369861731",
"TIC 15827263",
"TIC 610976842",
"TIC 24094603",
"TIC 232038798",
"TIC 1965256662",
"TIC 229400092",
"TIC 104866616",
"TIC 376740324",
"TIC 233602827",
"TIC 429302040",
"TIC 258777134",
"TIC 387690507",
"TIC 277987172",
"TIC 290348382",
"TIC 45684223",
"TIC 155986102",
"TIC 29830212",
"TIC 468997317",
"TIC 1717732429",
"TIC 286696848",
"TIC 219821826",
"TIC 271749758",
"TIC 139158696",
"TIC 160162987",
"TIC 1101909431",
"TIC 235678745",
"TIC 232982558",
"TIC 269701147",
"TIC 219854519",
"TIC 230086768",
"TIC 394029948",
"TIC 374908020",
"TIC 364075352",
"TIC 29781292",
"TIC 141488193",
"TIC 157698565",
"TIC 229742722",
"TIC 466206508",
"TIC 33878688",
"TIC 164652245",
"TIC 232608943",
"TIC 55525572",
"TIC 261257684",
"TIC 29781292",
"TIC 150151262",
"TIC 367855829",
"TIC 27769688",
"TIC 349095149",
"TIC 388130235",
"TIC 260128333",
"TIC 309792357",
"TIC 209464063",
"TIC 362220495",
"TIC 224313733",
"TIC 309402106",
"TIC 349576261",
"TIC 298663873",
"TIC 307958020",
"TIC 70887357",
"TIC 251086776",
"TIC 332630868",
"TIC 434100688",
"TIC 304142124",
"TIC 150151262",
"TIC 441739020",
"TIC 350618622",
"TIC 425163745",
"TIC 146434840",
"TIC 190430205",
"TIC 329145325",
"TIC 410528770",
"TIC 420288086",
"TIC 270333769"])]
# filter specific target
#tess_df = tess_df.loc[(tess_df['object_id'] == 'TIC 392476080')]
# filter hot hupiters
#tess_df = tess_df.loc[~((tess_df['type'] == 'planet_transit') & (tess_df['radius(earth)'] > 6) & (tess_df['period'] < 10))]
# filter non hot jupiters
#tess_df = tess_df.loc[((tess_df['type'] == 'planet_transit') & (tess_df['radius(earth)'] > 6) & (tess_df['period'] < 10))]
# results = iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", tess_df, batch_size, cores,
#                     thresholds=[0.5], calibrator_path="IATSON_planet_cal_isotonic.pkl",
#                     #validation_df=all_df,
#                     from_arrays=True, explain=False)
# results = results[0.5]
# results.loc[results['prediction_value'] < 1e-5, ['prediction_value']] = 1e-5
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"predictions_map_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )
# tps_above_lp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > lp_threshold))]
# tps_above_vp = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] > vp_threshold))]
# tps_above_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] >= 0.5))]
# fps_above_lp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold))]
# fps_above_vp = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] > vp_threshold))]
# fps_above_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] >= 0.5))]
# print(f"TPs above lp: {tps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_lp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above vp: {tps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_vp.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"TPs above 0.5: {tps_above_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {tps_above_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FPs above lp: {fps_above_lp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_lp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above vp: {fps_above_vp.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_vp.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FPs above 0.5: {fps_above_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {fps_above_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# tns_below_ln = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < ln_threshold))]
# tns_below_vn = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] < vn_threshold))]
# tns_below_05 = results.loc[((results['expected_class'] == 0) & (results['prediction_value'] <= 0.5))]
# fns_below_ln = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold))]
# fns_below_vn = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] < vn_threshold))]
# fns_below_05 = results.loc[((results['expected_class'] == 1) & (results['prediction_value'] <= 0.5))]
# print(f"TNs below ln: {tns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_ln.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below vn: {tns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_vn.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"TNs below 0.5: {tns_below_05.shape[0]} / {results.loc[results['expected_class'] == 0].shape[0]} = {tns_below_05.shape[0] / results.loc[results['expected_class'] == 0].shape[0]}")
# print(f"FNs below ln: {fns_below_ln.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_ln.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below vn: {fns_below_vn.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_vn.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# print(f"FNs below 0.5: {fns_below_05.shape[0]} / {results.loc[results['expected_class'] == 1].shape[0]} = {fns_below_05.shape[0] / results.loc[results['expected_class'] == 1].shape[0]}")
# results = results.loc[(((results['expected_class'] == 0) & (results['prediction_value'] > lp_threshold)) |
#                       ((results['expected_class'] == 1) & (results['prediction_value'] < ln_threshold)))]
# IATSON_planet.compute_statistics_map(results, f"{model_dir}./", filename=f"fps_fns_isotonic", markersize=60,
#                                      vp_threshold=vp_threshold, lp_threshold=lp_threshold,
#                                      vn_threshold=vn_threshold, ln_threshold=ln_threshold
#                                      )

# EXPLAIN TARGET
df = tess_df.loc[(tess_df['object_id'] == 'TIC 300013921')]
results = iatson.predict_df_cv(training_dir, f"{model_dir}/IATSON_planet_model_chk", df, batch_size, cores,
                    thresholds=[0.5], calibrator_path="IATSON_planet_cal_isotonic.pkl",
                    #validation_df=all_df,
                    from_arrays=True, explain=True)