import dataclasses
import logging
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas import DataFrame

from exoml.ete6.NanException import NanException
from exoml.iatson.iatson_planet_generator import IatsonPlanetModelGenerator
import matplotlib.pyplot as plt


@dataclasses.dataclass
class RowRegeneratorInput:
    df: DataFrame
    index: int
    plot_inputs: bool = False
    store_arrays: bool = True
    from_arrays: bool = False
    throw_nan_exception: bool = False

def row_regenerator(row_regenerator_input: RowRegeneratorInput):
    row = row_regenerator_input.df.iloc[row_regenerator_input.index]
    object_id = row['object_id']
    period = row['period']
    t0 = row['epoch']
    duration = row['duration(h)']
    type = row['type']
    object_id_tuple = object_id.split(' ')
    mission_id = object_id_tuple[0]
    target_id = int(object_id_tuple[1])
    lcs_dir = kic_dir if mission_id == 'KIC' else tic_dir
    file_prefix = lcs_dir + '/' + mission_id + '_' + str(target_id) + '_' + str(round(period, 2)) \
        if mission_id == 'KIC' else lcs_dir + '/' + str(mission_id) + ' ' + str(target_id) + '_' + str(round(period, 2))
    # if not os.path.exists(file_prefix + '_input_og.csv'):
    logging.info("Doing " + object_id + " " + str(row_regenerator_input.index) + " item of " + str(len(row_regenerator_input.df)) + '. Type=' + type + ', P=' +
                 str(np.round(period, 2)) + ', D=' + str(np.round(duration, 2)) + ', T0=' + str(np.round(t0, 2)))
    index = row_regenerator_input.df.index[(row_regenerator_input.df['object_id'] == object_id) &
                                   (row_regenerator_input.df['period'] == period)]
    img = IatsonPlanetModelGenerator(row_regenerator_input.df,
                                     data_dir, 1, [11, 135, 300, 75, 75, 75, 75, 75, 75],
                                     {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0],
                                      'tce_centroids_offset': [0],
                                      'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0],
                                      'tce_odd_even': [0], 'none': [0],
                                      'EB': [0], 'bckEB': [0], 'candidate': [0], 'tce_candidate': [0], 'fp_candidate': [0]},
                                     from_arrays=row_regenerator_input.from_arrays, store_arrays=row_regenerator_input.store_arrays,
                                     plot_inputs=row_regenerator_input.plot_inputs, shuffle_batch=False)
    try:
        batch_arrays, batch_tags = img.__getitem__(index[0])

        logging.info(f"Finished {index}")
        return np.array(batch_arrays[0]).flatten(), np.array(batch_arrays[1]).flatten(), np.array(batch_arrays[2]).flatten(), \
                np.array([np.mean(batch_arrays[3]), np.median(batch_arrays[3]), np.std(batch_arrays[3]), np.min(batch_arrays[3]), np.max(batch_arrays[3])]).flatten(), \
                np.array([np.mean(batch_arrays[4]), np.median(batch_arrays[4]), np.std(batch_arrays[4]), np.min(batch_arrays[4]), np.max(batch_arrays[4])]).flatten(), \
                np.array([np.mean(batch_arrays[5]), np.median(batch_arrays[5]), np.std(batch_arrays[5]), np.min(batch_arrays[5]), np.max(batch_arrays[5])]).flatten(), \
                np.array([np.mean(batch_arrays[6]), np.median(batch_arrays[6]), np.std(batch_arrays[6]), np.min(batch_arrays[6]), np.max(batch_arrays[6])]).flatten(), \
                np.array([np.mean(batch_arrays[7]), np.median(batch_arrays[7]), np.std(batch_arrays[7]), np.min(batch_arrays[7]), np.max(batch_arrays[7])]).flatten(), \
                np.array([np.mean(batch_arrays[8]), np.median(batch_arrays[8]), np.std(batch_arrays[8]), np.min(batch_arrays[8]), np.max(batch_arrays[8])]).flatten()
    except NanException as e:
        logging.info(f"Nans found for target {object_id}_{period}")
        return None
    except Exception as e:
        logging.exception(f"Error {index}")
        return None

def statistics(df, title, labels):
    cores = 1
    for label in labels:
        df_loop = df.reset_index(drop=True)
        # df_len = len(df_loop)
        # range_df = range(0, df_len)
        range_df = df_loop.index[df_loop['type'] == label].tolist()
        inputs = []
        for i in range_df:
            inputs = inputs + [RowRegeneratorInput(df_loop, i, plot_inputs=False, from_arrays=True, store_arrays=False, throw_nan_exception=True)]
        with Pool(processes=cores, maxtasksperchild=1) as pool:
            results = pool.map(row_regenerator, inputs, chunksize=1)
        scalar_values = np.zeros((12, len(results)))
        for index, result in enumerate(results):
            scalar_values[0][index] = result[0][0]
            scalar_values[1][index] = result[0][1]
            scalar_values[2][index] = result[0][2]
            scalar_values[3][index] = result[0][3]
            scalar_values[4][index] = result[0][4]
            scalar_values[5][index] = result[0][5]
            scalar_values[6][index] = result[0][6]
            scalar_values[7][index] = result[0][7]
            scalar_values[8][index] = result[0][8]
            scalar_values[9][index] = result[1][0]
            scalar_values[10][index] = result[2][0]
            scalar_values[11][index] = result[2][1]
        logging.info(f"-------{label} statistics-------")
        logging.info(f"Period-> mean: {np.mean(scalar_values[0])}, median: {np.median(scalar_values[0])}, "
                     f"std: {np.std(scalar_values[0])}, min: {np.min(scalar_values[0])}, "
                     f"max: {np.max(scalar_values[0])}")
        logging.info(f"Duration -> mean: {np.mean(scalar_values[1])}, median: {np.median(scalar_values[1])}, "
                     f"std: {np.std(scalar_values[1])}, min: {np.min(scalar_values[1])}, "
                     f"max: {np.max(scalar_values[1])}")
        logging.info(f"Re -> mean: {np.mean(scalar_values[2])}, median: {np.median(scalar_values[2])}, "
                     f"std: {np.std(scalar_values[2])}, min: {np.min(scalar_values[2])}, "
                     f"max: {np.max(scalar_values[2])}")
        logging.info(f"Gtn -> mean: {np.mean(scalar_values[3])}, median: {np.median(scalar_values[3])}, "
                     f"std: {np.std(scalar_values[3])}, min: {np.min(scalar_values[3])}, "
                     f"max: {np.max(scalar_values[3])}")
        logging.info(f"Gtr -> mean: {np.mean(scalar_values[4])}, median: {np.median(scalar_values[4])}, "
                     f"std: {np.std(scalar_values[4])}, min: {np.min(scalar_values[4])}, "
                     f"max: {np.max(scalar_values[4])}")
        logging.info(f"Bootstrap FAP -> mean: {np.mean(scalar_values[5])}, median: {np.median(scalar_values[5])}, "
                     f"std: {np.std(scalar_values[5])}, min: {np.min(scalar_values[5])}, "
                     f"max: {np.max(scalar_values[5])}")
        logging.info(f"Oe -> mean: {np.mean(scalar_values[6])}, median: {np.median(scalar_values[6])}, "
                     f"std: {np.std(scalar_values[6])}, min: {np.min(scalar_values[6])}, "
                     f"max: {np.max(scalar_values[6])}")
        logging.info(f"Temp diff -> mean: {np.mean(scalar_values[7])}, median: {np.median(scalar_values[7])}, "
                     f"std: {np.std(scalar_values[7])}, min: {np.min(scalar_values[7])}, "
                     f"max: {np.max(scalar_values[7])}")
        logging.info(f"Albedo -> mean: {np.mean(scalar_values[8])}, median: {np.median(scalar_values[8])}, "
                     f"std: {np.std(scalar_values[8])}, min: {np.min(scalar_values[9])}, "
                     f"max: {np.max(scalar_values[8])}")
        logging.info(f"Offset -> mean: {np.mean(scalar_values[9])}, median: {np.median(scalar_values[9])}, "
                     f"std: {np.std(scalar_values[9])}, min: {np.min(scalar_values[9])}, "
                     f"max: {np.max(scalar_values[9])}")
        logging.info(f"Teff -> mean: {np.mean(scalar_values[10])}, median: {np.median(scalar_values[10])}, "
                     f"std: {np.std(scalar_values[10])}, min: {np.min(scalar_values[10])}, "
                     f"max: {np.max(scalar_values[10])}")
        logging.info(f"Rstar -> mean: {np.mean(scalar_values[11])}, median: {np.median(scalar_values[11])}, "
                     f"std: {np.std(scalar_values[11])}, min: {np.min(scalar_values[11])}, "
                     f"max: {np.max(scalar_values[11])}")
        fig, axs = plt.subplots(4, 3, figsize=(20, 15), constrained_layout=True)
        axs[0][0].hist(scalar_values[0])
        axs[0][0].set_title("Period")
        axs[0][1].hist(scalar_values[1])
        axs[0][1].set_title("Duration")
        axs[0][2].hist(scalar_values[2])
        axs[0][2].set_title("Re")
        axs[1][0].hist(scalar_values[3])
        axs[1][0].set_title("Gtn")
        axs[1][1].hist(scalar_values[4])
        axs[1][1].set_title("Gtr")
        axs[1][2].hist(scalar_values[9])
        axs[1][2].set_title("Offset")
        axs[2][0].hist(scalar_values[5])
        axs[2][0].set_title("Bootstrap FAP")
        axs[2][1].hist(scalar_values[6])
        axs[2][1].set_title("Oe")
        axs[2][2].hist(scalar_values[7])
        axs[2][2].set_title("T_diff")
        axs[3][0].hist(scalar_values[8])
        axs[3][0].set_title("Albedo_diff")
        axs[3][1].hist(scalar_values[10])
        axs[3][1].set_title("Teff")
        axs[3][2].hist(scalar_values[11])
        axs[3][2].set_title("Rstar")
        fig.suptitle(f"{title}_{label}")
        plt.show()

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
 # while len(logger.handlers) > 0:
 #     logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

data_dir = "/data/scratch/ml/"
kic_dir = data_dir + "q1_q17/lcs"
# for filename in os.listdir(kic_dir):
#     if 'input' in filename:
#         logging.info("Removing %s", filename)
#         os.remove(kic_dir + '/' + filename)
tic_dir = data_dir + "ete6/lcs"
# for filename in os.listdir(tic_dir):
#     if 'input' in filename:
#         logging.info("Removing %s", filename)
#         os.remove(tic_dir + '/' + filename)

#df = pd.read_csv("/data/scratch/ml/q1_q17/injected_objects_tces.csv", index_col=False)
df1 = pd.read_csv(data_dir + "q1_q17/classified_tces.csv")
df2 = pd.read_csv(data_dir + "ete6/injected_objects.csv")
#df3 = pd.read_csv(data_dir + "q1_q17/injected_objects_candidates.csv")
# df = [df3]
# df = pd.concat(df)
#df = pd.merge(df1, df2, on=['object_id', 'period', 'epoch', 'duration(h)', 'type', 'depth_primary', 'radius(earth)', 'boot_fap'], how="outer")
# df = pd.merge(df1, df3, on=['object_id', 'period', 'epoch', 'duration(h)', 'type', 'depth_primary',
#                             'radius(earth)', 'boot_fap', 'st_rad', 'koi_srad', 'tce_sradius',
#                             'st_teff', 'koi_steff', 'tce_steff', 'pl_orbper', 'koi_period',
#                             'tce_period_x', 'tce_period_y', 'pl_tranmid', 'koi_time0bk',
#                             'tce_time0bk'], how="outer")
#df = df.loc[df['period'] > 30]
# df = df.loc[df['object_id'].isin([
# 'KIC 5724440'
# ])]
# df = df2
# df['boot_fap'] = np.nan
# df['st_rad'] = np.nan
# df['tce_sradius'] = np.nan
#range = df.index[df['object_id'] == 'KIC 9117123'].tolist()
#df = df.loc[(df['type'] == 'fp') & (df['tce_maxmes'] <= 7.1) & (df['cent_so_sigma'] <= 3) & (df['source_offset_sigma'] <= 3) & (df['kic_offset_sigma'] <= 3) & (df['tce_bin_oedp_stat'] <= 9) & (df['tce_hap_stat'] < df['tce_cap_stat'])]
#range = df.index[df['type'] == 'tce_secondary'].tolist()

cores = 1
df = df1
df = df.reset_index(drop=True)
df_len = len(df)
range_df = range(0, df_len)
range_df = df.index[df['object_id'] == 'KIC 9772531'].tolist()
inputs = []
for i in range_df:
    inputs = inputs + [RowRegeneratorInput(df, i, plot_inputs=True, from_arrays=False, store_arrays=False, throw_nan_exception=True)]
with Pool(processes=cores, maxtasksperchild=1) as pool:
    results = pool.map(row_regenerator, inputs, chunksize=1)
#statistics(df2, 'ete6', ['planet', 'none', 'EB', 'bckEB'])
statistics(df1, 'q1q17', ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
                        'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'tce_candidate', 'fp_candidate'])
