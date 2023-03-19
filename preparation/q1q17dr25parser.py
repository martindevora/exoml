import multiprocessing
from multiprocessing import Pool
import pandas as pd
import numpy as np

from preparation.parser_utils import CreateTargetInput, create_target_csv


def create_targets_df(tces_dir):
    # Merge TCEs DF and cumulative DF ignoring everything not from Kepler or K2
    # Mark TCEs as planet only if they have photometric and rv measurements
    # test_df = pd.read_csv(tces_dir + '/classified_tces.csv', comment='#')
    # test_df = test_df.sort_values(by=["object_id", "period"], ascending=True)
    # test_df = test_df[test_df['type'].isin(['fp', 'planet', 'planet_transit', 'tce'])]
    # test_df.to_csv(tces_dir + '/injected_objects_tces.csv')
    tces_df = pd.read_csv(tces_dir + '/q1_q17_dr25_tce_2022.09.13_14.23.37.csv', comment='#')
    tces_df['tce_period_round'] = tces_df['tce_period'].round(1)
    cum_df = pd.read_csv(tces_dir + '/cumulative_2022.09.13_14.24.29.csv', comment='#')
    cum_df['pl_name'] = cum_df['kepler_name']
    cum_df['tce_period'] = cum_df['koi_period']
    cum_df['tce_period_round'] = cum_df['tce_period'].round(1)
    tces_df = pd.merge(tces_df, cum_df, on=['kepid', 'tce_period_round'], how='outer')
    #disc_df = pd.read_csv(tces_dir + '/PSCompPars_2022.06.25_14.17.21.csv', comment='#')
    disc_df = pd.read_csv(tces_dir + '/PSCompPars_2022.09.13_13.37.12.csv', comment='#')
    disc_df = disc_df[disc_df['disc_facility'] == 'Kepler']
    disc_df = disc_df[disc_df['discoverymethod'] == 'Transit']
    disc_df = pd.merge(tces_df, disc_df, on='pl_name', how='outer')
    disc_df['type'] = 'tce'
    disc_df.loc[(disc_df['rv_flag'] > 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED'), 'type'] = 'planet'
    disc_df.loc[(disc_df['rv_flag'] == 0) & (disc_df['tran_flag'] > 0) & (disc_df['koi_disposition'] == 'CONFIRMED'), 'type'] = 'planet_transit'
    disc_df.loc[disc_df['koi_disposition'] == 'FALSE POSITIVE', 'type'] = 'fp'
    disc_df.loc[disc_df['koi_disposition'] == 'CANDIDATE', 'type'] = 'candidate'
    disc_df['object_id'] = 'KIC ' + disc_df['kepid'].astype('Int64').astype('str')
    disc_df['period'] = disc_df['tce_period_x']
    disc_df.loc[disc_df['period'].isna(), 'period'] = disc_df[disc_df['period'].isna()]['tce_period_y']
    disc_df.loc[disc_df['period'].isna(), 'period'] = disc_df[disc_df['period'].isna()]['pl_orbper']
    disc_df['epoch'] = disc_df['tce_time0bk']
    disc_df.loc[disc_df['epoch'].isna(), 'epoch'] = disc_df[disc_df['epoch'].isna()]['koi_time0bk']
    disc_df.loc[disc_df['epoch'].isna(), 'epoch'] = disc_df[disc_df['epoch'].isna()]['pl_tranmid'] - 2454833.0
    disc_df['duration(h)'] = disc_df['tce_duration']
    disc_df.loc[disc_df['duration(h)'].isna(), 'duration(h)'] = disc_df[disc_df['duration(h)'].isna()]['koi_duration']
    disc_df.loc[disc_df['duration(h)'].isna(), 'duration(h)'] = disc_df[disc_df['duration(h)'].isna()]['pl_trandur']
    disc_df = disc_df.sort_values(by=["object_id", "period"], ascending=True)
    last_target = ''
    last_period = 0
    last_index = -1
    left_match_index = []
    right_match_index = []
    unpaired_index = disc_df.index.values
    for index, target_row in disc_df.iterrows():
        if last_target == target_row['object_id'] and np.abs(target_row['period'] - last_period) < 0.05 * last_period:
            right_match_index = right_match_index + [index]
            left_match_index = left_match_index + [last_index]
            if last_index in unpaired_index:
                unpaired_index = unpaired_index[unpaired_index != last_index]
            if index in unpaired_index:
                unpaired_index = unpaired_index[unpaired_index != index]
        last_target = target_row['object_id']
        last_period = target_row['period']
        last_index = index
    #disc_df = disc_df.loc[left_match_index].combine_first(disc_df.loc[right_match_index])
    result_df = disc_df.loc[unpaired_index]
    for i, index in enumerate(left_match_index):
        result = disc_df.loc[index].combine_first(disc_df.loc[right_match_index[i]])
        result_df = pd.concat([result_df, result.to_frame().T], ignore_index=True)
    result_df.drop(result_df[result_df['duration(h)'].isnull()].index, inplace=True)
    #
    result_df.drop(result_df[result_df['object_id'] == 'KIC <NA>'].index, inplace=True)
    # Remove latest KIC from the catalogue
    result_df.drop(result_df[result_df['object_id'] == 'KIC 100001645'].index, inplace=True)
    result_df.to_csv(tces_dir + '/classified_tces.csv')
    #disc_df[disc_df['object_id'] == 'KIC 7377200'].iloc[0]['epoch']
    result_df = disc_df[disc_df['type'].isin(['fp', 'planet', 'planet_transit', 'tce'])]
    result_df = result_df[result_df['object_id'] != 'KIC 100001645'] #This KIC doesn't seem to exist
    #TODO this next statement is to prevent confirmed planets not matching any tce nor koi
    #result_df = result_df[~result_df['object_id'].str.contains('<NA')]
    result_df.to_csv(tces_dir + '/injected_objects_tces.csv')
    result_df = disc_df[disc_df['type'].isin(['fp', 'planet', 'planet_transit'])]
    result_df.to_csv(tces_dir + '/injected_objects.csv')


def create_target_csvs(csv, target_dir, cache_dir, cores=multiprocessing.cpu_count() - 1, force=False, ids=None):
    tces_df = pd.read_csv(csv, comment='#')
    tces_df = tces_df.sort_values(by=["kepid"], ascending=True)
    #tces_df = tces_df.drop_duplicates(subset=["kepid"], keep='last')
    tces_df = tces_df.sample(frac=1).reset_index(drop=True)
    print("Total number of targets is " + str(len(tces_df)))
    inputs = []
    if ids is not None:
        tces_df = tces_df[tces_df['object_id'].isin(ids)]
    for index, tce_row in tces_df.iterrows():
        inputs = inputs + [CreateTargetInput('KIC', tce_row['object_id'], None, target_dir, None, tce_row, cache_dir, force=force)]
    with Pool(processes=cores, maxtasksperchild=1) as pool:
        pool.map(create_target_csv, inputs, chunksize=1)
