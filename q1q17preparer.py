import logging
import os.path
import sys

from lcbuilder.star.KicStarCatalog import KicStarCatalog

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
df_dir = '/data/scratch/ml/tess/cps/'
os.makedirs(df_dir, exist_ok=True)
lcs_dir = df_dir + '/lcs/'
os.makedirs(lcs_dir, exist_ok=True)
targets_df = df_dir + '/tess_cps.csv'
synth_df = './exoml/resources/q1_q17/classified_tces_synthetic.csv'
cache_dir = home_dir
cores = 1
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')

df_dir = '/data/scratch/ml/q1_q17/'
targets_df = df_dir + 'classified_tces.csv'
lcs_dir = df_dir + '/lcs/'
create_targets_df(df_dir)
create_targets_multi_df(df_dir)
#create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), targets_df, cores, force=False)
# TESS CP catalog
# df_dir = '/data/scratch/ml/tess/cps/'
# targets_df = df_dir + '/tess_cps.csv'
# Q1Q17
df_dir = '/data/scratch/ml/q1_q17/'
targets_df = df_dir + 'classified_tces.csv'
cache_dir = home_dir
lcs_dir = df_dir + '/lcs/'
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