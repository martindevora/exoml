import logging
import multiprocessing
import sys
from multiprocessing import Pool
import pandas as pd
from lcbuilder.star.KicStarCatalog import KicStarCatalog

from preparation.parser_utils import CreateTargetInput, create_target_csv, create_star_csvs


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

def create_target_csvs(csv, target_dir, cache_dir, cores=multiprocessing.cpu_count() - 1):
    tces_df = pd.read_csv(csv, comment='#')
    tces_df = tces_df[tces_df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    tces_df = tces_df.sort_values(by=["kepid"], ascending=True)
    tces_df = tces_df.drop_duplicates(subset=["kepid"], keep='last')
    inputs = []
    for index, tce_row in tces_df.iterrows():
        inputs = inputs + [CreateTargetInput('KIC', tce_row['kepid'], None, target_dir, None, tce_row, cache_dir)]
    with Pool(processes=cores) as pool:
        pool.map(create_target_csv, inputs)



formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

home_dir = '/home/mdevora/'
root_dir = home_dir + '/scratch/ml/q1_q17/'
lcs_dir = root_dir + '/lcs/'
cache_dir = home_dir
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#_create_star_csv(CreateStarCsvInput('/home/mdevora/scratch/ml/q1_q17/lcs/KIC 12106934_time_series_long.csv'))
create_target_csvs(root_dir + '/classified_tces.csv', lcs_dir, cache_dir, 10)
create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), '*time_series_long*.csv', "KIC_([0-9]+)_time_series_long", 10)
#create_injection_dataframe(root_dir, lcs_dir)
#create_targets_df(root_dir)
#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
