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
    create_targets_multi_df

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

# TESS
df_dir = '/data/scratch/ml/tess/'
targets_df = df_dir + 'tess_tois.csv'
cache_dir = home_dir
lcs_dir = df_dir + '/lcs/'
create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['all'], mission_prefix='TIC',
                   ids=['TIC 55383975',
                        'TIC 99330090',
                        'TIC 54044474',
                        'TIC 289988797',
                        'TIC 363761216',
                        'TIC 272050287',
                        'TIC 229976631',
                        'TIC 376981340',
                        'TIC 170103435',
                        'TIC 90448944',
                        'TIC 379723086',
                        'TIC 271581073',
                        'TIC 219208037',
                        'TIC 69672493',
                        'TIC 302659412',
                        'TIC 382437043',
                        'TIC 169904935',
                        'TIC 231721005',
                        'TIC 290302097',
                        'TIC 151959065',
                        'TIC 289535142',
                        'TIC 169532369',
                        'TIC 357972447',
                        'TIC 71134735',
                        'TIC 446549905',
                        'TIC 316937670',
                        'TIC 268042363',
                        'TIC 143022742',
                        'TIC 294780517',
                        'TIC 255918903',
                        'TIC 350743714',
                        'TIC 89535068',
                        'TIC 288185138',
                        'TIC 177722855',
                        'TIC 366074069',
                        'TIC 19155785',
                        'TIC 333821572',
                        'TIC 259389219',
                        'TIC 356868487',
                        'TIC 470709084',
                        'TIC 168871868',
                        'TIC 220518305',
                        'TIC 320525204',
                        'TIC 86280613',
                        'TIC 461538935',
                        'TIC 84339983',
                        'TIC 100100827',
                        'TIC 15445551',
                        'TIC 399402994',
                        'TIC 137402055',
                        'TIC 99224131',
                        'TIC 261867566',
                        'TIC 441462736',
                        'TIC 36724087',
                        'TIC 234112540',
                        'TIC 439842788',
                        'TIC 112099249',
                        'TIC 94986319',
                        'TIC 201604954',
                        'TIC 229510866',
                        'TIC 257605131',
                        'TIC 17307715',
                        'TIC 92352620',
                        'TIC 418761354',
                        'TIC 355703913',
                        'TIC 192826603',
                        'TIC 139104940',
                        'TIC 36734222',
                        'TIC 218795833',
                        'TIC 38846515',
                        'TIC 149601126',
                        'TIC 290131778',
                        'TIC 178367144',
                        'TIC 39926974',
                        'TIC 209752908',
                        'TIC 254113311',
                        'TIC 374180079',
                        'TIC 358107516',
                        'TIC 358516596',
                        'TIC 399860444',
                        'TIC 234825296',
                        'TIC 69679391',
                        'TIC 371443216',
                        'TIC 102192004',
                        'TIC 104024556',
                        'TIC 374180079',
                        'TIC 183985250',
                        'TIC 36352297',
                        'TIC 283419843',
                        'TIC 311133118',
                        'TIC 435339558',
                        'TIC 33521996',
                        'TIC 14570099',
                        'TIC 268924036',
                        'TIC 271424111',
                        'TIC 468574941',
                        'TIC 470381900',
                        'TIC 391958006',
                        'TIC 405010127',
                        'TIC 120692092',
                        'TIC 386259537',
                        'TIC 27635334',
                        'TIC 308172249',
                        'TIC 27534138',
                        'TIC 915110182',
                        'TIC 25375553',
                        'TIC 139528693',
                        'TIC 77031414',
                        'TIC 27848472',
                        'TIC 369455629',
                        'TIC 46096489',
                        'TIC 55652896',
                        'TIC 17865622',
                        'TIC 380907135',
                        'TIC 101721385',
                        'TIC 301235044',
                        'TIC 428787891',
                        'TIC 138644215',
                        'TIC 271431851',
                        'TIC 375506058',
                        'TIC 305243225',
                        'TIC 454069765',
                        'TIC 16288184',
                        'TIC 52640302',
                        'TIC 452808876',
                        'TIC 281459670',
                        'TIC 112604564',
                        'TIC 230001847',
                        'TIC 169226822',
                        'TIC 284475976',
                        'TIC 278892590',
                        'TIC 461239485',
                        'TIC 256364928',
                        'TIC 1129033',
                        'TIC 258514800',
                        'TIC 399794420',
                        'TIC 374200604',
                        'TIC 390651552',
                        'TIC 455135327',
                        'TIC 366411016',
                        'TIC 198588220',
                        'TIC 366443426',
                        'TIC 144065872',
                        'TIC 27039476',
                        'TIC 201248411',
                        'TIC 32499655',
                        'TIC 382391899',
                        'TIC 155867025',
                        'TIC 236387002',
                        'TIC 352011875',
                        'TIC 67646988',
                        'TIC 346338552',
                        'TIC 191284318',
                        'TIC 374530847',
                        'TIC 292152376',
                        'TIC 147977348',
                        'TIC 393495970',
                        'TIC 183120439',
                        'TIC 377780790',
                        'TIC 381856447',
                        'TIC 47911178',
                        'TIC 61098812',
                        'TIC 257774438',
                        'TIC 162802770',
                        'TIC 38087018',
                        'TIC 200723869',
                        'TIC 354469661',
                        'TIC 166833457',
                        'TIC 36724087',
                        'TIC 271354351',
                        'TIC 293687315',
                        'TIC 62483237',
                        'TIC 380589029',
                        'TIC 162689341',
                        'TIC 142276270',
                        'TIC 437248515',
                        'TIC 409794137',
                        'TIC 264508014',
                        'TIC 231663901',
                        'TIC 366568760',
                        'TIC 336732616',
                        'TIC 183532609',
                        'TIC 306362738',
                        'TIC 138364831',
                        'TIC 237320326',
                        'TIC 445751830',
                        'TIC 266593143',
                        'TIC 14344979',
                        'TIC 348835438',
                        'TIC 273874849',
                        'TIC 240823272',
                        'TIC 158297421',
                        'TIC 32487566',
                        'TIC 49043968',
                        'TIC 32949762',
                        'TIC 395171208',
                        'TIC 459730973',
                        'TIC 291685334',
                        'TIC 332064670',
                        'TIC 402026209',
                        'TIC 149969625',
                        'TIC 243921117',
                        'TIC 8260536'])
# create_star_csvs(lcs_dir, 10, 20.25, TicStarCatalog(), targets_df, cores, force=False)

# check No data found for target


#create_tess_tces_df('/home/martin/Downloads/tess_tces/')
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
