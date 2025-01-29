import logging
import sys

from exoml.santo.santo_test_data_generator import SantoTestDataGenerator

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

SantoTestDataGenerator(output_dir="/data/scratch/ml/santo/training_data/",
                       curves_len_boundaries=(15000, 30000),
                       curves_count=10000,
                       curves_cadence=120,
                       white_noise_power_boundaries=(0.0001, 0.05),
                       red_noise_frequency_boundaries=(0.01, 10),
                       red_noise_power_boundaries=(0.0001, 0.05),
                       sinusoid_frequency_boundaries=(0.01, 10),
                       sinusoid_power_boundaries=(0.001, 0.05),
                       pseudo_periodic_frequency_boundaries=(0.01, 10),
                       pseudo_periodic_power_boundaries=(0.0001, 0.05),
                       pseudo_periodic_frequency_std_boundaries=(0.01, 0.1),
                       pseudo_periodic_power_std_boundaries=(0.0001, 0.001),
                       transit_depth_boundaries=(0.00005, 0.05),
                       transit_period_avg_std=(10, 9.5),
                       flare_amplitude_avg_std=(0.1, 0.05),
                       flare_length_avg_std=(0.2, 0.05),
                       max_number_of_pseudo_periodic=3,
                       max_number_of_sinusoid=3,
                       max_number_of_red_noise=5,
                       only_plot=False)