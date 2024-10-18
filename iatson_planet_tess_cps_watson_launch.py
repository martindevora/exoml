import gc
import logging
import os
import shutil

import numpy as np
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from watson.watson import Watson

from exoml.ml.model.base_model import HyperParams
from exoml.preparation.parser_utils import create_star_csv, CreateStarCsvInput
import astropy.units as u


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import pydevd_pycharm
#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
import pandas as pd
from exoml.iatson.IATSON_planet import IATSON_planet


if not isinstance(logging.root, logging.RootLogger):
    logging.root = logging.RootLogger(logging.INFO)

iatson = IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp', 3: 'fa', 4: 'tce', 5: 'tce_secondary',
                         6: 'tce_centroids_offset', 7: 'tce_source_offset', 8: 'tce_og', 9: 'tce_odd_even',
                         10: 'none', 11: 'EB', 12: 'bckEB'},
                        ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                         'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'none', 'EB', 'bckEB'],
                        {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'tce_centroids_offset': [0],
                         'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0], 'tce_odd_even': [0], 'none': [0],
                         'EB': [0], 'bckEB': [0], 'candidate': [0], 'tce_candidate': [0]}, HyperParams(), mode="all")\
     .build(use_transformers=False, transformer_blocks=1, transformer_heads=2)
tois_df = pd.read_csv('/home/administrator/.sherlockpipe/tois.csv')
tois_df = tois_df.loc[(tois_df["Disposition"] == "KP") | (tois_df["Disposition"] == "CP")]
habitability_calculator = HabitabilityCalculator()
predictions_df = pd.DataFrame(columns=['id', 'oi', 'predictions', 'predictions_cal', 'predictions_avg',
                                       'predictions_median', 'predictions_std', 'predictions_cal_avg',
                                       'predictions_cal_median', 'predictions_cal_std', 'exception'])
results_csv_file = './tess_cps_predictions.csv'
watson_base_dir = '/data/scratch/ml/watson/'
models_dir = '/home/administrator/exoml/07/'
if os.path.exists(results_csv_file):
    predictions_df = pd.read_csv(results_csv_file)
for index, row in tois_df.iterrows():
    target_id = row['Object Id']
    oi = row['OI']
    mode = 'first'
    try:
        predictions_for_oi = predictions_df.loc[(predictions_df['oi'] == oi)]
        if len(predictions_for_oi) > 0 and not predictions_for_oi['exception'].iloc[0]:
            logging.info("Skipping target %s as it has already been done", oi)
            continue
        elif len(predictions_for_oi) > 0 and predictions_for_oi['exception'].iloc[0]:
            predictions_df.drop(predictions_df[predictions_df['oi'] == oi].index)
            logging.info("Trying target %s as it was done but threw exception", oi)
            mode = 'repeat'
        else:
            logging.info("Trying target %s", oi)
        watson_dir = f'{watson_base_dir}/{oi}/'
        if os.path.exists(watson_dir):
            shutil.rmtree(watson_dir, ignore_errors=True)
        if not os.path.exists(watson_dir):
            os.makedirs(watson_dir, exist_ok=True)
        lc_filename = f'{watson_dir}/lc.csv'
        period = row['Period (days)']
        epoch = row['Epoch (BJD)'] - 2457000
        duration = row['Duration (hours)']
        depth_ppt = row['Depth (ppm)'] / 1000
        run_watson = False
        plot_inputs = True
        create_star_csv(CreateStarCsvInput(target_id, 4 if 'KIC' or 'EPIC' in target_id else 20.25, 50,
                                           KicStarCatalog() if 'KIC' in target_id else TicStarCatalog(),
                                           watson_dir, force=True))
        star_filename = f'{watson_dir}/{target_id}_star.csv'
        star_df = pd.read_csv(star_filename)
        a = habitability_calculator.calculate_semi_major_axis(period, star_df['mass'].iloc[0])
        a_rstar = LcbuilderHelper.convert_from_to(a, u.au, u.R_sun) / star_df['radius'].iloc[0]
        rp_rstar = LcbuilderHelper.convert_from_to(row['Planet Radius (R_Earth)'], u.R_earth, u.R_sun) / star_df['radius'].iloc[0]
        cadence = 'short'
        Watson(watson_dir, watson_dir).vetting(target_id, period, epoch, duration * 60,
                                                             depth_ppt, 'all', rp_rstar if not np.isnan(rp_rstar) else None,
                                                             a_rstar if not np.isnan(a_rstar) else None,
                                                             ra=star_df.iloc[0]['ra'], dec=star_df.iloc[0]['dec'],
                                                             cadence=cadence, cpus=4,
                                                             clean=True, iatson_enabled=False, iatson_inputs_save=False, only_summary=True)
                                                             #transits_mask=[{'P': 22.9134, 'D': 8 * 60, 'T0': 140.647}])
        star_filename = f'{watson_dir}/{target_id}_stars.csv'
        predictions, predictions_cal = iatson.predict_watson(target_id, period, duration, epoch, depth_ppt, watson_dir,
                                                             star_filename, lc_filename,
                                                             cv_dir=models_dir,
                                                             plot_inputs=plot_inputs, use_csvs=False)
                                                             #calibrator_path='/home/mdevo/exoml/models/iatson/final/IATSON_planet_cal_isotonic.pkl')
        if mode != 'repeat':
            predictions_df = pd.concat([predictions_df, pd.DataFrame.from_dict(
                            {'id': row['Object Id'], 'oi': row['OI'], 'predictions': [predictions],
                             'predictions_cal': [predictions_cal],
                             'predictions_avg': [np.nanmean(predictions)],
                             'predictions_median': [np.nanmedian(predictions)],
                             'predictions_std': [np.nanstd(predictions)],
                             'predictions_cal_avg': [np.nanmean(predictions_cal)],
                             'predictions_cal_median': [np.nanmedian(predictions_cal)],
                             'predictions_cal_std': [np.nanstd(predictions_cal)],
                             'exception': False},
                            orient='columns')], ignore_index=True)
        else:
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions'] = str(predictions)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_cal'] = str(predictions_cal)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_avg'] = np.nanmean(predictions)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_median'] = np.nanmedian(predictions)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_std'] = np.nanstd(predictions)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_cal_avg'] = np.nanmean(predictions_cal)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_cal_median'] = np.nanmedian(predictions_cal)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'predictions_cal_std'] = np.nanstd(predictions_cal)
            predictions_df.loc[predictions_df['oi'] == row['OI'], 'exception'] = False
        logging.info("Finished target %s", oi)
    except:
        logging.exception("Error for target %s", oi)
        if mode != 'repeat':
            predictions_df = pd.concat([predictions_df, pd.DataFrame.from_dict(
                {'id': row['Object Id'], 'oi': row['OI'], 'predictions': [np.nan],
                 'predictions_cal': [np.nan],
                 'predictions_avg': [np.nan],
                 'predictions_median': [np.nan],
                 'predictions_std': [np.nan],
                 'predictions_cal_avg': [np.nan],
                 'predictions_cal_median': [np.nan],
                 'predictions_cal_std': [np.nan],
                 'exception': True},
                orient='columns')], ignore_index=True)
    finally:
        logging.info(f"Collected {gc.collect()} objects")
    predictions_df.to_csv(results_csv_file, index=False)
