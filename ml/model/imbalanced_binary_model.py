from keras.metrics import BinaryAccuracy

from ml.loss.cross_entropy import binary_focal_loss
from ml.model.binary_model import BinaryModel


class ImbalancedBinaryModel(BinaryModel):
    def __init__(self, name, input_size, class_ids) -> None:
        super().__init__(name, input_size, class_ids)

    def instance_loss_accuracy(self):
        return binary_focal_loss, BinaryAccuracy()
