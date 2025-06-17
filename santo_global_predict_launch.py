import os
import random
import pandas as pd
import wotan
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lightkurve import TessLightCurve
import numpy as np

from exoml.santo.SANTOGlobal import SANTOGlobal
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
testing_data_dir = "/data/scratch/ml/santo/training_data/"
if not os.path.exists(testing_data_dir):
    os.mkdir(testing_data_dir)
white_noise_power = 0.001
target_files_dir = "/data/scratch/ml/santo/training_data/"
targets_files = sorted(os.listdir(target_files_dir))
targets_files = targets_files[0:5]
number_of_items = 1
items_in_batch = 1
batch_index = 0
santo = SANTOGlobal()
model_dir = "/home/administrator/exoml/SANTO_model_chk"
santo.predict(testing_data_dir, targets_files, model_dir,
              batch_size=1024, plot=True, tagged_data=True)
batch_index = batch_index + 1

##### Predict from TOI 540 (fast rotator)
target_id = "TIC 200322593" # fast rotator
# target_dir = "TIC_231663901_santo" # hot jupiter
# target_dir = "TIC_305048087_santo"
target_dir = f"/home/administrator/{target_id}_santo"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
lcbuild = MissionLightcurveBuilder().build(MissionObjectInfo([6], target_id, cadence=[120], author=['SPOC', 'TESS-SPOC'], high_rms_enabled=False),
                                                     target_dir, "/home/administrator/")
time = lcbuild.lc.time.value
flux = lcbuild.lc.flux.value
lc_df = pd.DataFrame(columns=['time', 'flux'])
lc_df['time'] = time
lc_df['flux'] = flux
lc_df.to_csv(f'{target_dir}/lc.csv')
SANTOGlobal().predict(target_dir, ['/lc.csv'],
                      model_dir,
                      batch_size=1024, plot=True, plot_positives=False)

target_id = "TIC 259377017" # toi 270
# target_dir = "TIC_231663901_santo" # hot jupiter
# target_dir = "TIC_305048087_santo"
target_dir = f"/home/administrator/{target_id}_santo"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
lcbuild = MissionLightcurveBuilder().build(MissionObjectInfo([5], target_id, cadence=[120], author=['SPOC', 'TESS-SPOC'], high_rms_enabled=False),
                                                     target_dir, "/home/administrator/")
time = lcbuild.lc.time.value
flux = lcbuild.lc.flux.value
lc_df = pd.DataFrame(columns=['time', 'flux'])
lc_df['time'] = time
lc_df['flux'] = flux
lc_df.to_csv(f'{target_dir}/lc.csv')
SANTOGlobal().predict(target_dir, ['/lc.csv'],
                      model_dir,
                      batch_size=1024, plot=True, plot_positives=False)
