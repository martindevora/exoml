import logging
import os.path
import shutil
import sys
import pandas as pd
import numpy as np
from lcbuilder.star.KicStarCatalog import KicStarCatalog
from lcbuilder.star.TicStarCatalog import TicStarCatalog

from exoml.preparation.parser_utils import create_star_csvs
from exoml.preparation.q1q17dr25parser import create_synth_targets_df, create_target_csvs, download_synth_targets, \
    create_synth_csvs, create_tess_cps_df, create_tess_fps_df, create_tess_tces_df, create_targets_df, \
    create_targets_multi_df, label_triage_tces_df, label_tces_df

if not isinstance(logging.root, logging.RootLogger):
    logging.root = logging.RootLogger(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

home_dir = '/home/administrator/'
# df_dir = '/data/scratch/ml/tess/cps/'
# os.makedirs(df_dir, exist_ok=True)
# lcs_dir = df_dir + '/lcs/'
# os.makedirs(lcs_dir, exist_ok=True)
# targets_df = df_dir + '/tess_cps.csv'
# synth_df = './exoml/resources/q1_q17/classified_tces_synthetic.csv'
# cache_dir = home_dir
cores = 1
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')

# df_dir = '/data/scratch/ml/q1_q17/'
# targets_df = df_dir + 'classified_tces.csv'
# lcs_dir = df_dir + '/lcs/'
# # Create Q1Q17 dataframes
# for fold in np.arange(0, 10):
#     model_dir = f"/home/administrator/exoml/IATSON_planet_model_chk_{fold}"
#     model_original_dir = f"/home/administrator/exoml/IATSON_planet_model_{fold}"
#     train_dir = f"{model_dir}/train_dataset.csv"
#     controversy_train_dir = f"{model_dir}/train_dataset_controversy.csv"
#     validation_dir = f"{model_dir}/validation_dataset.csv"
#     controversy_validation_dir = f"{model_dir}/validation_dataset_controversy.csv"
#     if not os.path.exists(controversy_train_dir):
#         shutil.copyfile(train_dir ,controversy_train_dir)
#     if not os.path.exists(controversy_validation_dir):
#         shutil.copyfile(validation_dir ,controversy_validation_dir)
#     training_df = pd.read_csv(train_dir)
#     val_df = pd.read_csv(validation_dir)
#     training_df = training_df.loc[~(((training_df['object_id'].str.startswith("KIC 11449844")) & (training_df['period'] > 38) & (training_df['period'] < 39)) |
#                                     ((training_df['object_id'].str.startswith("KIC 11974540")) & (training_df['period'] > 24) & (training_df['period'] < 25)) |
#                                     ((training_df['object_id'].str.startswith("KIC 7532973")) & (training_df['period'] > 2.1) & (training_df['period'] < 2.4)) |
#                                     ((training_df['object_id'].str.startswith("KIC 7532973")) & (training_df['period'] > 2.4) & (training_df['period'] < 2.6)) |
#                                     ((training_df['object_id'].str.startswith("KIC 8478994")) & (training_df['period'] > 51) & (training_df['period'] < 52)) |
#                                     ((training_df['object_id'].str.startswith("KIC 10585738")) & (training_df['period'] > 0.6) & (training_df['period'] < 0.7)) |
#                                     ((training_df['object_id'].str.startswith("KIC 6444896")) & (training_df['period'] > 19) & (training_df['period'] < 20)) |
#                                     ((training_df['object_id'].str.startswith("KIC 4742414")) & (training_df['period'] > 15) & (training_df['period'] < 16)) |
#                                     ((training_df['object_id'].str.startswith("KIC 4735826")) & (training_df['period'] > 11) & (training_df['period'] < 12)))]
#     val_df = val_df.loc[~(((val_df['object_id'].str.startswith("KIC 11449844")) & (val_df['period'] > 38) & (val_df['period'] < 39)) |
#                                     ((val_df['object_id'].str.startswith("KIC 11974540")) & (val_df['period'] > 24) & (val_df['period'] < 25)) |
#                                     ((val_df['object_id'].str.startswith("KIC 7532973")) & (val_df['period'] > 2.1) & (val_df['period'] < 2.4)) |
#                                     ((val_df['object_id'].str.startswith("KIC 7532973")) & (val_df['period'] > 2.4) & (val_df['period'] < 2.6)) |
#                                     ((val_df['object_id'].str.startswith("KIC 8478994")) & (val_df['period'] > 51) & (val_df['period'] < 52)) |
#                                     ((val_df['object_id'].str.startswith("KIC 10585738")) & (val_df['period'] > 0.6) & (val_df['period'] < 0.7)) |
#                                     ((val_df['object_id'].str.startswith("KIC 6444896")) & (val_df['period'] > 19) & (val_df['period'] < 20)) |
#                                     ((val_df['object_id'].str.startswith("KIC 4742414")) & (val_df['period'] > 15) & (val_df['period'] < 16)) |
#                                     ((val_df['object_id'].str.startswith("KIC 4735826")) & (val_df['period'] > 11) & (val_df['period'] < 12)))]
#     training_df.to_csv(train_dir, index=False)
#     training_df.to_csv(f"{model_original_dir}/train_dataset.csv", index=False)
#     val_df.to_csv(validation_dir, index=False)
#     val_df.to_csv(f"{model_original_dir}/validation_dataset.csv", index=False)


# create_targets_df(df_dir)
# create_targets_multi_df(df_dir)

#create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), targets_df, cores, force=False)
# TESS CP catalog
# df_dir = '/data/scratch/ml/tess/cps/'
# targets_df = df_dir + '/tess_cps.csv'

# Q1Q17
df_dir = '/data/scratch/ml/q1_q17/'
# targets_df = df_dir + 'classified_tces.csv'
# cache_dir = home_dir
# lcs_dir = df_dir + '/lcs/'
# df = pd.read_csv(df_dir + 'classified_tces_multi.csv')
# type_counts = df['multitype'].str.split(',').explode()
# type_counts = type_counts.value_counts()
# type_counts = type_counts[type_counts.index != '']
# combined_count = type_counts.get('tce', 0) + type_counts.get('tce_candidate', 0)
# type_counts = type_counts.drop(['tce', 'tce_candidate'], errors='ignore')
# type_counts['tce'] = combined_count
# positive_tags = type_counts[['planet_transit', 'planet']].dropna()
# candidate_tags = type_counts[['candidate', 'fp_candidate', 'planet_validated']].dropna()
# type_counts = type_counts.drop(['planet_transit', 'planet'], errors='ignore')
# type_counts = type_counts.drop(['candidate', 'fp_candidate', 'planet_validated'], errors='ignore')
# type_counts = type_counts.sort_values(ascending=False)
# type_counts = type_counts.append(candidate_tags)
# type_counts = type_counts.append(positive_tags)
# logging.info(type_counts)
# import matplotlib.pyplot as plt
# num_bars = len(type_counts)
# separator_pos = num_bars - 2.5  # Position between the last two and the rest
# plt.axvline(separator_pos, color='black', linestyle='--', linewidth=1)
# separator_pos = num_bars - 5.5  # Position between the last two and the rest
# plt.axvline(separator_pos, color='black', linestyle='--', linewidth=1)
# type_counts.plot(kind='bar', color='skyblue', figsize=(8, 5))
# plt.xlabel('Label', size=15)
# plt.ylabel('Number of samples', size=15)
# plt.xticks(rotation=45, ha='right', size=10)
# plt.tight_layout()
# plt.show()

# TESS TOIS
#df_dir = '/data/scratch/ml/tess/'
#targets_df = df_dir + 'tess_tois.csv'
#cache_dir = home_dir
#lcs_dir = df_dir + '/lcs/'
# create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['all'], mission_prefix='TIC')
# create_star_csvs(lcs_dir, 10, 20.25, TicStarCatalog(), targets_df, cores, force=False)

# check No data found for target

# TESS tces pre-processing and labeling
# create_tess_tces_df('/data/scratch/ml/tess/', single_sector_only=False)
# label_triage_tces_df('/data/scratch/ml/tess/')
# label_tces_df('/data/scratch/ml/tess/')

# TESS tces
df_dir = '/data/scratch/ml/tess/'
targets_df = df_dir + 'tces.csv'
cache_dir = home_dir
lcs_dir = df_dir + '/lcs_tces/'
create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['all'], mission_prefix='TIC', limit=10000)
# create_star_csvs(lcs_dir, 10, 20.25, TicStarCatalog(), targets_df, cores, force=False)


#create_tess_fps_df(df_dir)
#create_target_csvs('/home/martin/Downloads/tess_tces/tces.csv', '/home/martin/Downloads/tess_tces/lcs', cache_dir, cores, force=True, mode=['all'], mission_prefix='TIC')
#create_synth_targets_df(tces_dir="./exoml/resources/q1_q17/")
# download_synth_targets('/data/scratch/ml/q1_q17_synth/lcs/', ['/home/martin/Downloads/injected-light-curves-dr25-inj1.bat',
#                                                               '/home/martin/Downloads/injected-light-curves-dr25-inj2.bat',
#                                                               '/home/martin/Downloads/injected-light-curves-dr25-inj3.bat'], 'exoml/resources/q1_q17/classified_tces_synthetic.csv')
#create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['metrics'], mission_prefix='KIC')
# create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['og'], ids=[
#    'KIC 5724440' ,
#     'KIC 8123937'
# ])
#create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), targets_df, cores, force=False)
#create_synth_csvs(synth_df, lcs_dir, lcs_dir, cache_dir, cores=1, force=True, ids=['757280'])

#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
