import os
import random
import pandas as pd
import wotan
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
from lightkurve import TessLightCurve
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from exoml.santo.SANTO import SANTO
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##### Predict training data
target_files_dir = "/data/scratch/ml/santo/training_data/"
targets_files = os.listdir(target_files_dir)
number_of_items = 10
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
    batch_index = batch_index + 1
    santo.predict(times, fluxes, "SANTO_model_chk", batch_size=5000, tagged_data=tagged_data, plot=True, target_name=targets_files[batch_index])
##### Predict from real target
# target_id = "TIC 200322593" # fast rotator
#target_dir = "TIC_231663901_santo" # hot jupiter
#target_dir = "TIC_305048087_santo"
# target_dir = f"{target_id}_santo"
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
# lcbuild = MissionLightcurveBuilder().build(MissionObjectInfo('all', target_id, cadence='short', high_rms_enabled=False),
#                                                      target_dir, "/home/martin/")

#SANTO().predict([lcbuild.lc.time.value], [lcbuild.lc.flux.value], "SANTO_model_chk", batch_size=5000, tagged_data=[tagged_data])
#SANTO().predict([lcbuild.lc.time.value], [lcbuild.lc.flux.value], "SANTO_model_chk", batch_size=5000, plot=True, plot_positives=False)
#SANTO().predict([lcbuild.lc.time.value], [wotan.flatten(lcbuild.lc.time.value, lcbuild.lc.flux.value, window_length=0.35)], "SANTO_model_chk", batch_size=5000)
