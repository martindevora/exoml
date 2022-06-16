from keras.callbacks import Callback


class MetricsPlotCallback(Callback):

    def __init__(self, base_model, model_dir, steps_per_epoch):
        super().__init__()
        self.base_model = base_model
        self.model_dir = model_dir
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.base_model.plot_metrics(self.model_dir, self.steps_per_epoch, with_validation=True)
