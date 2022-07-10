from keras.metrics import CategoricalAccuracy
from ml.loss.cross_entropy import categorical_focal_loss
from ml.model.base_model import BaseModel


class ImbalancedCategoricalModel(BaseModel):
    def __init__(self, class_names, name, input_size, class_ids) -> None:
        super().__init__(name, input_size, class_ids)
        self.class_names = class_names

    def instance_loss_accuracy(self):
        return categorical_focal_loss, CategoricalAccuracy()
