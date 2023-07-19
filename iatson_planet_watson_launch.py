import logging
import os
import numpy as np
from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from watson.watson import Watson

from ml.model.base_model import HyperParams
from preparation.parser_utils import create_star_csv, CreateStarCsvInput

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#import pydevd_pycharm
#pydevd_pycharm.settrace('38.red-83-57-176.dynamicip.rima-tde.net', port=5678, stdoutToServer=True, stderrToServer=True)

import os
import pandas as pd
from iatson.IATSON_planet import IATSON_planet


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
target_id = "TIC 282576340"
watson_dir = f'/home/mdevo/exoml/watson/{target_id}/'
if not os.path.exists(watson_dir):
    os.makedirs(watson_dir, exist_ok=True)
star_filename = f'{watson_dir}/{target_id}_star.csv'
lc_filename = f'{watson_dir}/lc.csv'
period = 8.3761316
epoch = 1210.732023
duration = 1.976
depth_ppt = 9060 / 1000
a_rstar = 0.1
rp_rstar = 0.1
run_watson = True
plot_inputs = True
if run_watson:
    create_star_csv(CreateStarCsvInput(target_id, 4 if 'KIC' or 'EPIC' in target_id else 20.25, 50,
                                       KicStarCatalog() if 'KIC' in target_id else TicStarCatalog(),
                                       watson_dir, force=False))
    star_df = pd.read_csv(star_filename)
    Watson(f'{watson_dir}', f'{watson_dir}').vetting(target_id, period, epoch, duration * 60,
                                                         depth_ppt, 'all', rp_rstar, a_rstar,
                                                         ra=star_df.iloc[0]['ra'], dec=star_df.iloc[0]['dec'],
                                                         cadence='short', cpus=os.cpu_count() // 2,
                                                         clean=False)
                                                         #transits_mask=[{'P': 22.9134, 'D': 8 * 60, 'T0': 140.647}])
    create_star_csv(CreateStarCsvInput(target_id, 4 if 'KIC' or 'EPIC' in target_id else 20.25, 50,
                                           KicStarCatalog() if 'KIC' in target_id else TicStarCatalog(),
                                           watson_dir, force=False))
predictions, predictions_cal = iatson.predict_watson(target_id, period, duration, epoch, depth_ppt, watson_dir,
                                                     star_filename, lc_filename,
                                                     cv_dir="/home/mdevo/exoml/models/iatson/final/",
                                                     plot_inputs=plot_inputs,
                                                     calibrator_path='/home/mdevo/exoml/models/iatson/final/IATSON_planet_cal_isotonic.pkl')
print(f'Uncalibrated predictions are {predictions}')
print(f'Calibrated predictions are {predictions_cal}')
print(f'Average uncalibrated prediction: {np.nanmean(predictions)} +- {np.nanstd(predictions)}')
print(f'Average calibrated prediction: {np.nanmean(predictions_cal)} +- {np.nanstd(predictions_cal)}')
