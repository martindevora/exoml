import os
import random
import pandas as pd
import wotan
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lightkurve import TessLightCurve
import numpy as np

from exoml.santo.santo_test_data_generator import SantoTestDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from exoml.santo.SANTO import SANTO
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##### Predict training data
# target_files_dir = "/data/scratch/ml/santo/training_data/"
# targets_files = os.listdir(target_files_dir)
# number_of_items = 30
# items_in_batch = 1
# batch_index = 0
# santo = SANTO()
# for index in np.arange(0, number_of_items):
#     times = []
#     fluxes = []
#     tagged_data = []
#     for batch_index in np.arange(batch_index, batch_index + items_in_batch):
#         target_data = np.loadtxt(target_files_dir + targets_files[batch_index], delimiter=',')
#         times = times + [np.linspace(0, len(target_data[0]) * 120 / 3600 / 24, len(target_data[0]))]
#         fluxes = fluxes + [target_data[0]]
#         tagged_data = tagged_data + [target_data[1]]
#     santo.predict(times, fluxes, "/home/administrator/exoml/SANTO_model_chk", batch_size=5000, tagged_data=tagged_data, plot=True, target_name=targets_files[batch_index])
#     batch_index = batch_index + 1

##### Predict from specific simulated target
testing_data_dir = "/data/scratch/ml/santo/testing_data/"
if not os.path.exists(testing_data_dir):
    os.mkdir(testing_data_dir)
white_noise_power = 0.001
SantoTestDataGenerator(output_dir="/data/scratch/ml/santo/testing_data/",
                       curves_len_boundaries=(15000, 30000),
                       curves_count=1,
                       curves_cadence=120,
                       white_noise_power_boundaries=(white_noise_power, white_noise_power),
                       red_noise_frequency_boundaries=(0.5, 0.5),
                       red_noise_power_boundaries=(white_noise_power * 0.1, white_noise_power * 0.1),
                       sinusoid_frequency_boundaries=(1.0, 1.0),
                       sinusoid_power_boundaries=(0.0002, 0.0002),
                       pseudo_periodic_frequency_boundaries=(0.0001, 0.0001),
                       pseudo_periodic_power_boundaries=(0.0001, 0.0001),
                       pseudo_periodic_frequency_std_boundaries=(0.000001, 0.000001),
                       pseudo_periodic_power_std_boundaries=(0.0000001, 0.0000001),
                       transit_depth_boundaries=(white_noise_power * 5, white_noise_power * 5),
                       transit_period_avg_std=(10, 9.5),
                       flare_amplitude_avg_std=(0.1, 0.05),
                       flare_length_avg_std=(0.2, 0.05),
                       max_number_of_pseudo_periodic=3,
                       max_number_of_sinusoid=3,
                       max_number_of_red_noise=5,
                       only_plot=False)
target_files_dir = "/data/scratch/ml/santo/testing_data/"
targets_files = os.listdir(target_files_dir)
number_of_items = 1
items_in_batch = 1
batch_index = 0
santo = SANTO()
for index in np.arange(0, number_of_items):
    times = []
    fluxes = []
    tagged_data = []
    for batch_index in np.arange(batch_index, batch_index + items_in_batch):
        target_data = np.loadtxt(target_files_dir + targets_files[batch_index], delimiter=',')
        times = times + [np.linspace(0, len(target_data[0]) * 120 / 3600 / 24, len(target_data[0]))]
        fluxes = fluxes + [target_data[0]]
        tagged_data = tagged_data + [target_data[1]]
    santo.predict(times, fluxes, "/home/administrator/exoml/SANTO_model_chk", batch_size=5000, tagged_data=tagged_data, plot=True, target_name=targets_files[batch_index])
    batch_index = batch_index + 1

##### Predict from real target
# target_id = "TIC 200322593" # fast rotator
# # target_dir = "TIC_231663901_santo" # hot jupiter
# # target_dir = "TIC_305048087_santo"
# target_dir = f"{target_id}_santo"
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
# lcbuild = MissionLightcurveBuilder().build(MissionObjectInfo('all', target_id, cadence=[120], author=['SPOC', 'TESS-SPOC'], high_rms_enabled=False),
#                                                      target_dir, "/home/administrator/")
#
# #SANTO().predict([lcbuild.lc.time.value], [lcbuild.lc.flux.value], "SANTO_model_chk", batch_size=5000, tagged_data=[tagged_data])
# SANTO().predict([lcbuild.lc.time.value], [lcbuild.lc.flux.value], "/home/administrator/exoml/SANTO_model_chk", batch_size=5000, plot=True, plot_positives=False)
# #SANTO().predict([lcbuild.lc.time.value], [wotan.flatten(lcbuild.lc.time.value, lcbuild.lc.flux.value, window_length=0.2)], "SANTO_model_chk", batch_size=5000, plot=True, plot_positives=False)
