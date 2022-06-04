from keras.metrics import Precision, Recall, AUC

from transpot.TRANSPOT import TRANSPOT


training_dir = '/mnt/DATA-2/ete6'
batch_size = 5
epochs = 50
model_dir = './'
transpot = TRANSPOT().build()#.inform(model_dir)
transpot.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=0.001,
               gradient_clip_norm=0.1, dry_run=False)
transpot = TRANSPOT().load_model(model_dir)
transpot.predict_test_set(training_dir, './test_dataset.csv', batch_size)

