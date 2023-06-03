import pandas as pd
import os
import numpy as np
import datetime


folds = 10
model_dir = './IATSON_planet_model'
if folds is not None:
    for fold in np.arange(0, folds):
        path = model_dir + f'_{fold}' + '/training_log.csv'
        log_df = pd.read_csv(path)
        print(f"R@P99_{fold}={log_df['val_r@p99'].max()}. Obtained in {datetime.datetime.fromtimestamp(os.stat(path).st_mtime)}")
else:
    path = model_dir + '/training_log.csv'
    log_df = pd.read_csv(path)
    print(f"R@P99={log_df['val_r@p99'].max()}. Obtained in {datetime.datetime.fromtimestamp(os.stat(path).st_mtime)}")