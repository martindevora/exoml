import os

from iatson.IATSON import IATSON

#tf.config.run_functions_eagerly(True)
# thread_config.setup_threads(1)

training_dir = '/mnt/DATA-2/ete6'
batch_size = 200
epochs = 50
model_dir = './'
iatson = IATSON(['bckeb', 'eb', 'planet', 'none']).build()#.inform(model_dir)
iatson.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=0.01, gradient_clip_norm=0.1,
             dry_run=False, cores=os.cpu_count() - 1)
iatson = IATSON().load_model(model_dir)
iatson.predict_test_set(training_dir, model_dir, batch_size)
