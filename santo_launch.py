import ml
from santo.SANTO import SANTO

lcs_dir = '/data/scratch/ml/ete6/lcs/'
injected_objects_file = '/data/scratch/ml/ete6/injected_objects.csv'
hp = ml.model.base_model.HyperParams(dropout_rate=0.1)
SANTO().train(lcs_dir, injected_objects_file, './SANTO', 50, hp)
