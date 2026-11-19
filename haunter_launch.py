import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from exoml.haunter.haunter import Haunter
from exoml.ml.model.base_model import HyperParams, BaseModel

from exoml.ml.callback.checkpoint import ModelCheckpointCallback

training_dir = os.environ.get('HAUNTER_TRAINING_DIR', '/home/martin/workspace/ph/exoml/exoml/resources/haunter/')
model_dir = os.environ.get('HAUNTER_MODEL_DIR', training_dir)

# the simulations are deterministic: no dropout or input noise, a wide residual network and a cosine schedule
batch_size = 1024
initial_lr = 0.002
epochs = 100
train_percent = 0.9
validation_percent = 0.1
steps_per_epoch = int(len(Haunter.load_data(training_dir)) * (1 - validation_percent)) // batch_size
hp = HyperParams(batch_size=batch_size,
                 epochs=epochs,
                 initial_learning_rate=initial_lr,
                 learning_rate_schedule=keras.optimizers.schedules.CosineDecay(initial_lr, steps_per_epoch * epochs, alpha=1e-3),
                 train_percent=train_percent,
                 validation_percent=validation_percent,
                 gradient_clip_norm=1.0,
                 gradient_clip_value=None,
                 dry_run=False,
                 early_stopping_delta=0,
                 early_stopping_patience=0,
                 custom_loss=None,
                 run_eagerly=False,
                 dropout_rate=0.0,
                 white_noise_std=0.0,
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
haunter.train(training_dir, model_dir, hp)
