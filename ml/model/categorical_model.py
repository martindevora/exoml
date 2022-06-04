from keras.metrics import CategoricalAccuracy
from ml.loss.cross_entropy import cross_entropy_with_dca_loss
from ml.model.base_model import BaseModel


class CategoricalModel(BaseModel):
    def __init__(self, name, input_size) -> None:
        super().__init__(name, input_size)

    def instance_loss_accuracy(self):
        return cross_entropy_with_dca_loss, CategoricalAccuracy()
