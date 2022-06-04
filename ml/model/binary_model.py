from keras.metrics import Precision, Recall, AUC, BinaryAccuracy

from ml.loss.cross_entropy import binary_cross_entropy_with_dca_loss
from ml.model.base_model import BaseModel


class BinaryModel(BaseModel):
    def __init__(self, name, input_size) -> None:
        super().__init__(name, input_size)

    def instance_loss_accuracy(self):
        return binary_cross_entropy_with_dca_loss, BinaryAccuracy()

    def instance_metrics(self):
        return [Precision(), Recall(), AUC(name="roc")]
