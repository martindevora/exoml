from transpot.TRANSPOT import TRANSPOT


training_dir = '/mnt/DATA-2/ete6'
batch_size = 5
epochs = 50
model_dir = './'
lr = 0.001
transpot = TRANSPOT({0: 'none', 1: ['bckEB', 'EB', 'planet']}).build()#.inform(model_dir)
transpot.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=lr,
               gradient_clip_norm=0.1, dry_run=False, balance_class_id=1)
#transpot = TRANSPOT().load_model(model_dir)
#transpot.predict_test_set(training_dir, './test_dataset.csv', batch_size)

