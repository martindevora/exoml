import logging

import numpy as np


class ModelWeightsLogger:
    def log_model_weights(self, model):
        # this function runs at the end of each epoch
        logging.info('Generating layer weights info')
        # loop over each layer and get weights and biases
        for layer_i in range(len(model.layers)):
            weights_and_bias = model.layers[layer_i].get_weights()
            if len(weights_and_bias) > 0:
                w = weights_and_bias[0]
                b = weights_and_bias[1]
                logging.debug('Layer no %s of type %s has weights of shape %s and biases of shape %s' % (
                    layer_i, model.layers[layer_i], np.shape(w), np.shape(b)))
                nan_weight_indexes = np.argwhere(np.isnan(w)).flatten()
                nan_bias_indexes = np.argwhere(np.isnan(w)).flatten()
                nan_weight_indexes_count = len(nan_weight_indexes)
                nan_bias_indexes_count = len(nan_bias_indexes)
                logging.debug('Layer max weight is ' + str(np.nanmax(w)) + ' and min weight is ' + str(np.nanmin(w)))
                logging.debug('Layer max bias is ' + str(np.nanmax(b)) + ' and min bias is ' + str(np.nanmin(b)))
                if nan_weight_indexes_count > 0 or nan_bias_indexes_count > 0:
                    logging.warning('Layer ' + str(layer_i) + ' has ' + str(nan_weight_indexes_count) +
                          ' nan weights and ' + str(nan_bias_indexes_count) + ' nan biases')
