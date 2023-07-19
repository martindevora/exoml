import logging

import pandas as pd
import os
import numpy as np
import datetime

from ml.model.base_model import BaseModel

logging.basicConfig(level=logging.DEBUG)
folds = 10
model_dir = './IATSON_planet_model'
class_ids = {0: ['planet_transit'], 1: ['planet'], 2: ['fp'], 3: ['fa'], 4: ['tce'],
                                         5: ['tce_secondary'],
                                         6: ['tce_centroids_offset'], 7: ['tce_source_offset'], 8: ['tce_og'],
                                         9: ['tce_odd_even'],
                                         10: ['none'], 11: ['EB'], 12: ['bckEB']}
if folds is not None:
    for fold in np.arange(0, folds):
        logging.info(f"DISTRIBUTIONS FOR K-{fold}")
        train_path = model_dir + f'_{fold}' + '/train_dataset.csv'
        val_path = model_dir + f'_{fold}' + '/validation_dataset.csv'
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        # BaseModel.log_sets_distribution(train_df, val_df, len(train_df) + len(val_df),
        #                                 class_ids=class_ids
        #                                 )
        for label in class_ids.values():
            train_rows = train_df[train_df['type'] == label[0]]
            val_rows = val_df[val_df['type'] == label[0]]
            total_train_labels = len(train_rows)
            total_val_labels = len(val_rows)
            kic_train_labels = len(train_rows[train_rows['object_id'].str.contains("KIC")])
            tic_train_labels = total_train_labels - kic_train_labels
            kic_val_labels = len(val_rows[val_rows['object_id'].str.contains("KIC")])
            tic_val_labels = total_val_labels - kic_val_labels
            logging.info(f"Label {label[0]}: Train {total_train_labels}, Train KIC {kic_train_labels},"
                         f"Val {total_val_labels}, Val KIC {kic_val_labels}")
else:
    path = model_dir + '/training_log.csv'
    log_df = pd.read_csv(path)
    print(f"R@P99={log_df['val_r@p99'].max()}. Obtained in {datetime.datetime.fromtimestamp(os.stat(path).st_mtime)}")