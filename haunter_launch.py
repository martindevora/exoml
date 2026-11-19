import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from exoml.haunter.haunter import Haunter
from exoml.ml.model.base_model import HyperParams, BaseModel

from exoml.ml.callback.checkpoint import ModelCheckpointCallback

training_dir = '/home/martin/workspace/ph/exoml/exoml/resources/haunter/'
model_dir = '/home/martin/workspace/ph/exoml/exoml/resources/haunter/'
results_path = f'{model_dir}/hyperparam_search/results.json'

# Try to load best hyperparams from search
# try:
#     with open(results_path) as f:
#         results = json.load(f)
#     valid = [r for r in results if 'val_mae_radius' in r]
#     valid.sort(key=lambda r: r['val_mae_radius'])
#     best = valid[0]
#     print(f'Loaded best from search: val_mae_radius={best["val_mae_radius"]:.4f}')
#     batch_size = best['batch_size']
#     initial_lr = best['initial_learning_rate']
#     dropout_rate = best['dropout_rate']
#     learning_rate_decay = best['learning_rate_decay']
#     white_noise_std = best['white_noise_std']
# except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as e:
#     print(f'No search results found ({e}), using defaults')
# the simulations are deterministic: no dropout or input noise, a wide residual network and a long cosine schedule
batch_size = 1024
initial_lr = 0.002
dropout_rate = 0.0
learning_rate_decay = 0.995
white_noise_std = 0.0

epochs = 100
hp = HyperParams(batch_size=batch_size,
            epochs=epochs,
            initial_learning_rate=initial_lr,
            train_percent=0.9,
            gradient_clip_norm=1.0,
            gradient_clip_value=None,
            dry_run=False,
            learning_rate_decay=learning_rate_decay,
            early_stopping_delta=0,
            early_stopping_patience=0,
            custom_loss=None,
            run_eagerly=False,
            dropout_rate=dropout_rate,
            white_noise_std=white_noise_std,
                 callbacks=[
                     ModelCheckpointCallback(filepath=f"{model_dir}/Haunter_model/Haunter_model_chk" + BaseModel.MODEL_SAVE_EXTENSION,
                                             original_path=f"{model_dir}/Haunter_model",
                                             verbose=0, save_weights_only=False,
                                             save_best_only=True, monitor='val_mae_radius',
                                             mode='min', save_freq='epoch',
                                             initial_value_threshold=0.004)  # dex: 1% of the radius
                 ]
                 )
haunter = Haunter(hp, width=384, blocks=3).build().inform(model_dir)
#haunter.train(training_dir, model_dir, hp)
haunter.train_cv(training_dir, model_dir, hp, folds=10)
