import logging
import sys
import os
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

dir = "/data/scratch/ml/q1_q17/lcs"
for filename in os.listdir(dir):
    if 'input' in filename:
        logging.info("Removing %s", filename)
        os.remove(dir + '/' + filename)

#df = pd.read_csv("/data/scratch/ml/q1_q17/injected_objects_tces.csv", index_col=False)
df = pd.read_csv("/data/scratch/ml/q1_q17/classified_tces.csv", index_col=False)
#df = shuffle(df)
#df.reset_index(inplace=True, drop=True)
img = IatsonPlanetModelGenerator(df, dir, 1, [11, 135, 2500, 500, 500, 500, 500, 500, 500],
                                 {'fp': [0], 'planet': [1], 'planet_transit': [1], 'tce': [0], 'candidate': [0],
                                  'none': [0]}, from_arrays=False, store_arrays=True, plot_inputs=False)
df_len = len(df)
for i in range(0, df_len):
    object_id = df.iloc[i]['object_id']
    logging.info("Doing " + object_id + " " + str(i) + " item of " + str(df_len))
    img.__getitem__(i)
