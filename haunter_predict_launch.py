import os
import pandas as pd

from exoml.ml.callback.checkpoint import ModelCheckpointCallback
from exoml.ml.callback.learning_rate import WarmUpAndLinDecreaseCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from exoml.haunter.haunter import Haunter
#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from exoml.ml.model.base_model import HyperParams

training_dir = '/home/martin/workspace/ph/exoml/exoml/resources/haunter/'
model_dir = '/home/martin/workspace/ph/exoml/exoml/resources/haunter/'
initial_lr = 0.001
final_lr = 0.0001
warmup_lr = 1e-7
epochs = 2000
warmup_epochs = 50
hp = HyperParams(batch_size=128,
            epochs=200,
            initial_learning_rate=initial_lr,
            gradient_clip_norm=0.5,
            gradient_clip_value=0.1,
            dry_run=False,
            learning_rate_decay=0.98,
            early_stopping_delta=0,
            early_stopping_patience=0,
            custom_loss=None,
            run_eagerly=True,
            dropout_rate=0.15,
            white_noise_std=0.001,
                 callbacks=[
                     # ReduceLROnPlateau(monitor='val_pr_auc', mode='max', factor=0.75, patience=10, min_lr=lr / 100), #, #,
                     ModelCheckpointCallback(filepath=f"{model_dir}/Haunter_model/Haunter_model_chk.keras",
                                             original_path=f"{model_dir}/Haunter_model",
                                             verbose=0, save_weights_only=False,
                                             save_best_only=True, monitor='val_r2_total',
                                             mode='max', save_freq='epoch',
                                             initial_value_threshold=0.9)
                     #WarmUpAndLinDecreaseCallback(initial_lr, initial_lr, warmup_lr, warmup_epochs, epochs - warmup_epochs)]
                 ])
haunter = Haunter(hp).build().inform(model_dir)
#haunter.load_model(model_dir + '/Haunter_model/Haunter_model_chk.keras')
#dataset = haunter.predict_validation(model_dir + '/Haunter_model', training_dir)
#dataset = haunter.predict_test(model_dir + '/Haunter_model', model_dir + '/test_set.csv')
haunter.predict_test_cv(model_dir, model_dir + '/test_set.csv', folds=10)
# haunter.plot_hists(dataset, additional_columns=['radius_Rearth_output', 'radius_Rearth_output_norm',
#                                       'radius_Rearth_output_pred', 'radius_Rearth_output_norm_pred',
#                                       'radius_Rearth_output_pred_diff', 'radius_Rearth_output_pred_diff_ratio',
#                                       'entropy_SI_output', 'entropy_SI_output_norm',
#                                       'entropy_SI_output_pred', 'entropy_SI_output_norm_pred',
#                                       'entropy_SI_output_pred_diff', 'entropy_SI_output_pred_diff_ratio',
#                                       'f_s_SI_output', 'f_s_SI_output_norm',
#                                       # 'f_s_SI_output_pred', 'f_s_SI_output_norm_pred',
#                                       # 'f_s_SI_output_pred_diff', 'f_s_SI_output_pred_diff_ratio',
#                                       'log10_mass_integral_output', 'log10_mass_integral_output_norm',
#                                       'log10_mass_integral_output_pred', 'log10_mass_integral_output_norm_pred',
#                                       'log10_mass_integral_output_pred_diff', 'log10_mass_integral_output_pred_diff_ratio',])
