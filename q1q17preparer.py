import logging
import sys

from lcbuilder.star.KicStarCatalog import KicStarCatalog

from preparation.parser_utils import create_star_csvs
from preparation.q1q17dr25parser import create_target_csvs, create_targets_df, create_targets_multi_df

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

home_dir = '/home/mdevo/'
df_dir = '/data/scratch/ml/q1_q17/'
lcs_dir = df_dir + '/lcs/'
targets_df = df_dir + '/classified_tces.csv'
cache_dir = home_dir
cores = 10
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_targets_df(df_dir)
#create_targets_multi_df(df_dir)
create_target_csvs(targets_df, lcs_dir, cache_dir, cores, force=True, mode=['all'], ids=['KIC 7203326'])
#create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), targets_df, cores, force=False)

#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))