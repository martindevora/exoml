import logging
import sys
from exoml.preparation.q1q17dr25parser import create_target_csvs, create_targets_df

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
cache_dir = home_dir
cores = 4
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_star_csv(CreateStarCsvInput('/home/mdevora/scratch/ml/q1_q17/lcs/KIC_100001645_lc_long.csv', 20.25, 6.5, "(KIC)_([0-9]+)_lc_long", KicStarCatalog(), lcs_dir))
#create_targets_df(df_dir)
create_target_csvs(df_dir + '/classified_tces.csv', lcs_dir, cache_dir, cores, force=True)
#create_star_csvs(lcs_dir, 50, 4, KicStarCatalog(), '*lc_long*.csv', "(KIC)_([0-9]+)_lc_long", cores)

#_create_target_csv(CreateTargetInput(lcs_dir + '/tess2019128220341-0000000230382377-0016-s_lc.fits', root_dir, pd.read_csv(root_dir + '/injected_objects.csv')))
