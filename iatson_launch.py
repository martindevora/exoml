import os

from iatson.IATSON import IATSON

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)

training_dir = '/mnt/DATA-2/ete6'
batch_size = 100
epochs = 50
model_dir = './'
cores = 0
lr = 0.01
iatson = IATSON({0: 'bckEB', 1: 'EB', 2: 'planet', 3: 'none'}, ['bckEB', 'EB', 'planet', 'none']).build()#.inform(model_dir)
#iatson.load_model("./IATSON_model")
iatson.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=lr, gradient_clip_norm=0.1,
             dry_run=False, cores=cores, balance_class_id=2)
#iatson = IATSON().load_model(model_dir)
#iatson.predict_test_set(training_dir, model_dir, batch_size)
