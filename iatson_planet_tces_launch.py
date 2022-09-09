import os

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)
from iatson.IATSON_planet import IATSON_planet
from ml.loss.cross_entropy import f1_loss
from ml.model.base_model import HyperParams

training_dir = '/mnt/DATA-2/q1_q17'
model_dir = './'
hp = HyperParams(batch_size=100,
            epochs=75,
            initial_learning_rate=0.001,
            gradient_clip_norm=0.1,
            gradient_clip_value=0.5,
            dry_run=False,
            cores=0,
            balance_class_sampling=[1, 1, 1],
            learning_rate_decay=0.98,
            early_stopping_delta=0,
            early_stopping_patience=0,
            custom_loss=None,
            run_eagerly=False)
iatson = IATSON_planet({0: 'planet_transit', 1: 'planet', 2: 'fp'},
                       ['planet_transit', 'planet', 'tp'],
                       {'fp': [0], 'planet': [1], 'planet_transit': [1]})\
    .build(use_transformers=False, transformer_blocks=3, transformer_heads=3)
#iatson.load_model("./IATSON_planet_model_bck")
iatson.train(training_dir, model_dir, hp)
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, "./IATSON_planet_model_bck", batch_size, os.cpu_count() - 1)
