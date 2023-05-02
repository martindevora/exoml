import dataclasses
import logging
import sys
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas import DataFrame

from iatson.iatson_planet_generator import IatsonPlanetModelGenerator


@dataclasses.dataclass
class RowRegeneratorInput:
    df: DataFrame
    index: int

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
    logging.info("Doing " + object_id + " " + str(row_regenerator_input.index) + " item of " + str(df_len) + '. Type=' + type + ', P=' +
                 str(np.round(period, 2)) + ', D=' + str(np.round(duration, 2)) + ', T0=' + str(np.round(t0, 2)))
    img = IatsonPlanetModelGenerator(row_regenerator_input.df[(row_regenerator_input.df['object_id'] == object_id) &
                                                              (row_regenerator_input.df['period'] == period)],
                                     data_dir, 1, [11, 135, 300, 75, 75, 75, 75, 75, 75],
                                     {'fp': [0], 'fa': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0],
                                      'tce_centroids_offset': [0],
                                      'tce_source_offset': [0], 'tce_secondary': [0], 'tce_og': [0],
                                      'tce_odd_even': [0], 'none': [0],
                                      'EB': [0], 'bckEB': [0]}, from_arrays=False, store_arrays=True,
                                     plot_inputs=False, shuffle_batch=False)
    img.__getitem__(0)


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
df1 = pd.read_csv(data_dir + "q1_q17/injected_objects_tces.csv", usecols=['object_id', 'period', 'epoch', 'duration(h)', 'type', 'depth_primary', 'radius(earth)'])
df2 = pd.read_csv(data_dir + "ete6/injected_objects.csv", usecols=['object_id', 'period', 'epoch', 'duration(h)', 'type', 'depth_primary', 'radius(earth)'])
df = [df2]
df = pd.concat(df)
cores = 10
df_len = len(df)
inputs = []
df = df.reset_index(drop=True)
range = range(0, df_len)
# range = df.index[df['object_id'] == 'KIC 5962532'].tolist()
for i in range:
    inputs = inputs + [RowRegeneratorInput(df, i)]
with Pool(processes=cores, maxtasksperchild=1) as pool:
    pool.map(row_regenerator, inputs, chunksize=1)
