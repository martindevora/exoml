import gzip
import logging
import multiprocessing
import os
import sys
import pathlib
import shutil
import traceback
from multiprocessing import Pool
import astropy.units as u
import pandas as pd
import numpy as np
import wotan
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from lcbuilder.star.TicStarCatalog import TicStarCatalog
from lightkurve import TessLightCurve
from numpy.random import default_rng
from ml.physics.transit_functions import t14
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder


def create_targets_df(tces_dir):
    # Merge TCEs DF and cumulative DF ignoring everything not from Kepler or K2
    # Mark TCEs as planet only if they have photometric and rv measurements
    tces_df = pd.read_csv(tces_dir + '/q1_q17_dr25_tce_2022.06.22_13.49.14.csv', comment='#')
    tces_df['tce_period'] = tces_df['tce_period'].round(2)
    cum_df = pd.read_csv(tces_dir + '/cumulative_2022.05.31_02.29.20.csv', comment='#')
    cum_df['pl_name'] = cum_df['kepler_name']
    cum_df['tce_period'] = cum_df['koi_period']
    cum_df['tce_period'] = cum_df['tce_period'].round(2)
    tces_df = pd.merge(tces_df, cum_df, on=['kepid', 'tce_period'], how='outer')
    disc_df = pd.read_csv(tces_dir + '/PSCompPars_2022.06.25_14.17.21.csv', comment='#')
    disc_df = disc_df[disc_df['disc_facility'] == 'Kepler']
    disc_df = disc_df[disc_df['discoverymethod'] == 'Transit']
    disc_df = pd.merge(tces_df, disc_df, on='pl_name', how='outer')
    disc_df['type'] = 'tce'
    disc_df[(disc_df['rv_flag'] > 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED')]['type'] = 'planet'
    disc_df[(disc_df['rv_flag'] == 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED')]['type'] = 'planet_transit'
    disc_df[disc_df['koi_disposition'] == 'FALSE POSITIVE']['type'] = 'fp'
    disc_df[disc_df['koi_disposition'] == 'CANDIDATE']['type'] = 'candidate'
    disc_df.to_csv(tces_dir + '/classified_tces.csv')

def prepare_labeled_lcs(csv, target_dir, cache_dir):
    tces_df = pd.read_csv(csv, comment='#')
    tces_df = tces_df[tces_df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    failed_targets_short = []
    failed_targets_long = []
    tces_df = tces_df.sort_values(by=["kepid"], ascending=True)
    mission_lightcurve_builder = MissionLightcurveBuilder()
    mission_ffi_lightcurve_builder = MissionFfiLightcurveBuilder()
    for index, tce_row in tces_df.iterrows():
        mission_id = 'KIC ' + str(int(tce_row['kepid']))
        try:
            logging.info("Trying to get short cadence info for %s", mission_id)
            file_name = target_dir + mission_id + "_time_series_short.csv"
            if not os.path.exists(file_name):
                lcbuild_short = \
                    mission_lightcurve_builder.build(MissionObjectInfo(mission_id, 'all'), target_dir, cache_dir)
                lc_short = lcbuild_short.lc
                lcbuild_short.lc_data.to_csv(file_name)
        except Exception as e:
            logging.exception("Failed short cadence for %s", mission_id)
            failed_targets_short = failed_targets_short + [mission_id]
        try:
            logging.info("Trying to get long cadence info for %s", mission_id)
            file_name = target_dir + mission_id + "_time_series_long.csv"
            if not os.path.exists(file_name):
                lcbuild_long = \
                    mission_ffi_lightcurve_builder.build(MissionFfiIdObjectInfo(mission_id, 'all'), target_dir, cache_dir)
                sectors = lcbuild_long.sectors
                lc_long = lcbuild_long.lc
                lcbuild_long.lc_data.to_csv(file_name)
            #logging.info("Downloading neighbour stars for %s" + mission_id)
            #TODO get neighbours light curves stars = self.download_neighbours(prepare_tic_input.tic, sectors)
            # TODO store folded light curves -with local and global views-(masking previous candidates?)
        except Exception as e:
            logging.exception("Failed long cadence for %s", mission_id)
            failed_targets_long = failed_targets_long + [mission_id]
        if index % 100:
            shutil.rmtree(cache_dir + "/.lightkurve-cache", ignore_errors=True)
            logging.info("Failed short targets %.0f", len(failed_targets_short))
            logging.info("Failed long targets %.0f", len(failed_targets_long))
    np.savetxt(target_dir + 'failed_targets_short.csv', np.array(failed_targets_short), delimiter=',', fmt='%f')
    np.savetxt(target_dir + 'failed_targets_long.csv', np.array(failed_targets_long), delimiter=',', fmt='%f')


formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

root_dir = '/home/mdevora/scratch/ml/q1_q17/'
lcs_dir = root_dir + 'lcs/'
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_star_csvs(lcs_dir)
#create_injection_dataframe(root_dir, lcs_dir)
#create_targets_df(root_dir)
prepare_labeled_lcs(root_dir + '/classified_tces.csv', target_dir=lcs_dir,
                    cache_dir="/home/mdevora/scratch/")
#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
