from transpot.TRANSPOT import TRANSPOT
import tensorflow as tf

training_dir = '/home/mdevora/scratch/ml/ete6/'
batch_size = 5
epochs = 50
model_dir = './'
transpot = TRANSPOT().build()#.inform(model_dir)
transpot.train(training_dir, model_dir, batch_size, epochs, initial_learning_rate=0.001,
               gradient_clip_norm=0.1, dry_run=False, metrics=[tf.keras.metrics.AUC(num_thresholds=200,
            curve='ROC',
            summation_method='interpolation',
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=False,
            num_labels=1,
            label_weights=None,
            from_logits=False)])
transpot = TRANSPOT().load_model(model_dir)
transpot.predict_test_set(training_dir, './test_dataset.csv', batch_size)

