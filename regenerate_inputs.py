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
            try:
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
            except:
                logging.info("Error")
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
ete6_dir = data_dir + "ete6/lcs"
# for filename in os.listdir(tic_dir):
#     if 'input' in filename:
#         logging.info("Removing %s", filename)
#         os.remove(tic_dir + '/' + filename)

#df = pd.read_csv("/data/scratch/ml/q1_q17/injected_objects_tces.csv", index_col=False)
df1 = pd.read_csv(data_dir + "q1_q17/classified_tces.csv")
df2 = pd.read_csv(data_dir + "ete6/injected_objects.csv")
#df3 = pd.read_csv(data_dir + "q1_q17/injected_objects_candidates.csv")
df4 = pd.read_csv(data_dir + "tess/tess_tois.csv")
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
df = df4
df = df.reset_index(drop=True)
df_len = len(df)
range_df = range(0, df_len)
range_df = [8143]
#range_df = df.index[(df['object_id'] == 'KIC 8480285') & (df['period'] < 30) & (df['period'] > 28)].tolist()
range_df = df.index[df['object_id'].isin(['TIC 380619414'
        'TIC 138359318',
        'TIC 14570099',
        'TIC 158324245',
        'TIC 164786087',
        'TIC 166527623',
        'TIC 329148988',
        'TIC 404243877',
        'TIC 112604564',
        'TIC 97735908',
        'TIC 144700903',
        'TIC 164173105',
        'TIC 251848941',
        'TIC 363548415',
        'TIC 97409519',
        'TIC 142276270',
        'TIC 175236511',
        'TIC 35516889',
        'TIC 366443426',
        'TIC 27491137',
        'TIC 16740101',
        'TIC 424865156',
        'TIC 34068865',
        'TIC 466840711',
        'TIC 392476080',
        'TIC 467971286',
        'TIC 339522221',
        'TIC 459730973',
        'TIC 134537478',
        'TIC 116264089',
        'TIC 129979528',
        'TIC 365007485',
        'TIC 22529346',
        'TIC 259506033',
        'TIC 207077681',
        'TIC 267572272',
        'TIC 257774438',
        'TIC 7088246',
        'TIC 37718056',
        'TIC 257527578',
        'TIC 150096001',
        'TIC 55652896',
        'TIC 427761355',
        'TIC 48217457',
        'TIC 112099249',
        'TIC 142276270',
        'TIC 232568235',
        'TIC 236445129',
        'TIC 14614418',
        'TIC 434226736',
        'TIC 100566492',
        'TIC 86396382',
        'TIC 55652896',
        'TIC 4610830',
        'TIC 436575927',
        'TIC 330690135',
        'TIC 432549364',
        'TIC 293435336',
        'TIC 240681314',
        'TIC 354619337',
        'TIC 178284730',
        'TIC 388811663',
        'TIC 55092869',
        'TIC 284475976',
        'TIC 245392284',
        'TIC 27534138',
        'TIC 375942197',
        'TIC 176868951'])]

# range_df = df.index[~df['pl_tranmid'].isna()]

# range_df = df.index[~df['object_id'].isin(["TIC 9837861",
# "TIC 114103136",
# "TIC 303867090",
# "TIC 365683032",
# "TIC 233059608",
# "TIC 349790953",
# "TIC 241102583",
# "TIC 124709665",
# "TIC 374847873",
# "TIC 244089109",
# "TIC 269450900",
# "TIC 317520667",
# "TIC 233602827",
# "TIC 367366318",
# "TIC 242460075",
# "TIC 98286422",
# "TIC 110178537",
# "TIC 305512837",
# "TIC 20892672",
# "TIC 167227214",
# "TIC 194179013",
# "TIC 16102323",
# "TIC 350153977",
# "TIC 23863106",
# "TIC 325670253",
# "TIC 183979262",
# "TIC 331034236",
# "TIC 387527558",
# "TIC 257060897",
# "TIC 284326455",
# "TIC 369861731",
# "TIC 15827263",
# "TIC 610976842",
# "TIC 24094603",
# "TIC 232038798",
# "TIC 1965256662",
# "TIC 229400092",
# "TIC 104866616",
# "TIC 376740324",
# "TIC 233602827",
# "TIC 429302040",
# "TIC 258777134",
# "TIC 387690507",
# "TIC 277987172",
# "TIC 290348382",
# "TIC 45684223",
# "TIC 155986102",
# "TIC 29830212",
# "TIC 468997317",
# "TIC 1717732429",
# "TIC 286696848",
# "TIC 219821826",
# "TIC 271749758",
# "TIC 139158696",
# "TIC 160162987",
# "TIC 1101909431",
# "TIC 235678745",
# "TIC 232982558",
# "TIC 269701147",
# "TIC 219854519",
# "TIC 230086768",
# "TIC 394029948",
# "TIC 374908020",
# "TIC 364075352",
# "TIC 29781292",
# "TIC 141488193",
# "TIC 157698565",
# "TIC 229742722",
# "TIC 466206508",
# "TIC 33878688",
# "TIC 164652245",
# "TIC 232608943",
# "TIC 55525572",
# "TIC 261257684",
# "TIC 29781292",
# "TIC 150151262",
# "TIC 367855829",
# "TIC 27769688",
# "TIC 349095149",
# "TIC 388130235",
# "TIC 260128333",
# "TIC 309792357",
# "TIC 209464063",
# "TIC 362220495",
# "TIC 224313733",
# "TIC 309402106",
# "TIC 349576261",
# "TIC 298663873",
# "TIC 307958020",
# "TIC 70887357",
# "TIC 251086776",
# "TIC 332630868",
# "TIC 434100688",
# "TIC 304142124",
# "TIC 150151262",
# "TIC 441739020",
# "TIC 350618622"])]


#range_df = df.index[(df['object_id'] == 'KIC 11017901')].tolist()
inputs = []
with Pool(processes=cores, maxtasksperchild=1) as pool:
    for i in range_df:
        inputs = inputs + [RowRegeneratorInput(df, i, plot_inputs=False, from_arrays=False, store_arrays=True, throw_nan_exception=True)]
    results = pool.map(row_regenerator, inputs, chunksize=1)
# for index, input in enumerate(inputs):
#     print(input.df.iloc[input.index]['object_id'] + ' ' + str(results[index][0][5]))
# statistics(df2, 'ete6', ['planet', 'none', 'EB', 'bckEB'])
# statistics(df1, 'q1q17', ['planet_transit', 'planet', 'fp', 'fa', 'tce', 'tce_og',
#                         'tce_secondary', 'tce_source_offset', 'tce_centroids_offset', 'tce_odd_even', 'tce_candidate', 'fp_candidate'])
