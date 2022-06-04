from keras.metrics import CategoricalAccuracy, Precision, Recall, AUC

from ml.loss.cross_entropy import categorical_cross_entropy_with_dca_loss
from ml.model.base_model import BaseModel


class CategoricalModel(BaseModel):
    def __init__(self, class_names, name, input_size) -> None:
        super().__init__(name, input_size)
        self.class_names = class_names

    def instance_loss_accuracy(self):
        return categorical_cross_entropy_with_dca_loss, CategoricalAccuracy()

    def instance_metrics(self):
        metrics = []
        for index, class_name in enumerate(self.class_names):
            class_name_string = str(class_name)
            metrics = metrics + [Precision(class_id=index, name=class_name_string + "_precision"),
                                 Recall(class_id=index, name=class_name_string + "_recall")]
        return metrics + [AUC(multi_label=True, num_labels=len(self.class_names), name="roc")]
