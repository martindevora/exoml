import logging
import sys

from lcbuilder.star.TicStarCatalog import TicStarCatalog

from exoml.preparation.ete6parser import create_target_csvs, create_injection_dataframe

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

root_dir = '/data/scratch/ml/ete6/'
lcs_dir = root_dir + 'lcs/'
targets_df = root_dir + '/injected_objects.csv'
cores = 1
#uncompress_data('/mnt/DATA-2/ete6/backebs/BackEBs/')
#uncompress_data('/mnt/DATA-2/ete6/ebs/EBs/')
#uncompress_data('/mnt/DATA-2/ete6/planets/Planets/')
#create_injection_dataframe(root_dir, lcs_dir)
#create_star_csvs(lcs_dir, 50, 4, TicStarCatalog(), targets_df, cores, force=False, ids=['TIC 318948330'])
create_target_csvs(lcs_dir, root_dir, cores=cores, force=True, mode=['metrics'],
                   ids=['TIC 9625658'])

# import matplotlib.pyplot as plt
# from foldedleastsquares import fold
#
# time_folded = fold(time, period, 1004.844738 + period / 2)
# indexes = np.argsort(time_folded)
#
# plt.scatter(time_folded[indexes], flux[indexes])
# plt.xlim(0.5 - duration / period * 2, 0.5 + duration / period * 2)
# plt.show()
