import logging
import sys
import os
import numpy as np
import pandas as pd
from iatson.iatson_planet_generator import IatsonPlanetModelGenerator


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
df1 = pd.read_csv(data_dir + "q1_q17/injected_objects_tces.csv", usecols=['object_id', 'period', 'epoch', 'duration(h)', 'type'])
df2 = pd.read_csv(data_dir + "ete6/injected_objects.csv", usecols=['object_id', 'period', 'epoch', 'duration(h)', 'type'])
df = [df2, df1]
df = pd.concat(df)
#df = shuffle(df)
#df.reset_index(inplace=True, drop=True)
#df = df[df['object_id'] == 'TIC 101832931']
img = IatsonPlanetModelGenerator(df, data_dir, 1, [11, 135, 2500, 500, 500, 500, 500, 500, 500],
                                 {'fp': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'candidate': [0],
                                  'none': [0], 'EB': [0], 'bckEB': [0]}, from_arrays=False, store_arrays=True,
                                 plot_inputs=False)
df_len = len(df)
for i in range(0, df_len):
    row = df.iloc[i]
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
    #if not os.path.exists(file_prefix + '_input_og.csv'):
    logging.info("Doing " + object_id + " " + str(i) + " item of " + str(df_len) + '. Type= ' + type + ', P=' +
                 str(np.round(period, 2)) + ', D=' + str(np.round(duration, 2)) + ', T0=' + str(np.round(t0, 2)))
    #if type not in ['none', 'fp']:
    img.__getitem__(i)
