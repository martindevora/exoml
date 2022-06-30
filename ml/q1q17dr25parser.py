import gzip
import multiprocessing
import os
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


root_dir = '/mnt/DATA-2/q1_q17/'
lcs_dir = root_dir + 'lcs/'
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_star_csvs(lcs_dir)
#create_injection_dataframe(root_dir, lcs_dir)
create_targets_df(root_dir)
#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
