import logging
import os
import numpy as np
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from watson.watson import Watson
import astropy.units as u
from exoml.ml.model.base_model import HyperParams
from exoml.preparation.parser_utils import CreateStarCsvInput, create_star_csv

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
target_id = "KIC 11449844"
watson_dir = f'./watson/{target_id}/'
if not os.path.exists(watson_dir):
    os.makedirs(watson_dir, exist_ok=True)
star_filename = f'{watson_dir}/{target_id}_star.csv'
lc_filename = f'{watson_dir}/lc.csv'
candidates_df = pd.read_csv('/data/scratch/ml/q1_q17/classified_tces.csv')
candidates_df = candidates_df.loc[candidates_df['kepid'] == 11449844]
period = 38.4788
epoch = 151.853
duration = 5.05845
run_watson = False
plot_inputs = True
if run_watson:
    create_star_csv(CreateStarCsvInput(target_id, 4 if 'KIC' or 'EPIC' in target_id else 20.25, 50,
                                       KicStarCatalog() if 'KIC' in target_id else TicStarCatalog(),
                                       watson_dir, force=True))
    star_df = pd.read_csv(star_filename)
    depth_ppt = LcbuilderHelper.convert_from_to(candidates_df.iloc[0]['radius(earth)'], u.R_earth, u.R_sun) ** 2 / star_df.iloc[0]['radius'] ** 2 * 1000
    radius_planet = np.sqrt(depth_ppt / 1000 * star_df.iloc[0]['radius'] ** 2)
    rp_rstar = radius_planet / star_df.iloc[0]['radius']
    a_rstar = LcbuilderHelper.convert_from_to(HabitabilityCalculator().calculate_semi_major_axis(period, star_df.iloc[0]['mass']), u.au, u.R_sun) / star_df.iloc[0]['radius']
    Watson(f'{watson_dir}', f'{watson_dir}').vetting(target_id, period, epoch, duration * 60,
                                                         depth_ppt, 'all', rp_rstar, a_rstar,
                                                         ra=star_df.iloc[0]['ra'], dec=star_df.iloc[0]['dec'],
                                                         cadence='short', cpus=os.cpu_count() // 2,
                                                         clean=False)
                                                         #transits_mask=[{'P': 22.9134, 'D': 8 * 60, 'T0': 140.647}])
    create_star_csv(CreateStarCsvInput(target_id, 4 if 'KIC' or 'EPIC' in target_id else 20.25, 50,
                                           KicStarCatalog() if 'KIC' in target_id else TicStarCatalog(),
                                           watson_dir, force=False))
else:
    star_df = pd.read_csv(star_filename)
    depth_ppt = LcbuilderHelper.convert_from_to(candidates_df.iloc[0]['radius(earth)'], u.R_earth, u.R_sun) ** 2 / star_df.iloc[0]['radius'] ** 2 * 1000
    radius_planet = np.sqrt(depth_ppt / 1000 * star_df.iloc[0]['radius'] ** 2)
    rp_rstar = radius_planet / star_df.iloc[0]['radius']
    a_rstar = LcbuilderHelper.convert_from_to(HabitabilityCalculator().calculate_semi_major_axis(period, star_df.iloc[0]['mass']), u.au, u.R_sun) / star_df.iloc[0]['radius']
predictions, predictions_cal = iatson.predict_watson(target_id, period, duration, epoch, depth_ppt, watson_dir,
                                                     star_filename, lc_filename,
                                                     cv_dir="/home/martin/.watson/0.0.4/",
                                                     plot_inputs=plot_inputs, batch_size=50, explain=False)
                                                     #calibrator_path='/home/mdevo/exoml/models/iatson/final/IATSON_planet_cal_isotonic.pkl')
print(f'Uncalibrated predictions are {predictions}')
print(f'Calibrated predictions are {predictions_cal}')
print(f'Average uncalibrated prediction: {np.nanmean(predictions)} +- {np.nanstd(predictions)}')
print(f'Median uncalibrated prediction: {np.nanmedian(predictions)} +- {np.nanstd(predictions)}')
print(f'Average calibrated prediction: {np.nanmean(predictions_cal)} +- {np.nanstd(predictions_cal)}')
print(f'Median calibrated prediction: {np.nanmedian(predictions_cal)} +- {np.nanstd(predictions_cal)}')
