import logging
import sys

from preparation.ete6parser import create_target_csvs, create_injection_dataframe

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
while len(logger.handlers) > 0:
    logger.handlers.pop()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

root_dir = '/data/scratch/ml/ete6/'
lcs_dir = root_dir + 'lcs/'
cores = 1
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_star_csvs(lcs_dir, 10, 20.25, TicStarCatalog(), '*_lc.csv', "([0-9]+)_lc.csv", cores)
#create_injection_dataframe(root_dir, lcs_dir)
create_target_csvs(lcs_dir, root_dir, cores=cores, force=True)
