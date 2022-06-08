from keras.metrics import CategoricalAccuracy, Precision, Recall, AUC, RecallAtPrecision

from ml.loss.cross_entropy import categorical_focal_loss
from ml.model.base_model import BaseModel


class CategoricalModel(BaseModel):
    def __init__(self, class_names, name, input_size) -> None:
        super().__init__(name, input_size)
        self.class_names = class_names

    def instance_loss_accuracy(self):
        return categorical_focal_loss, CategoricalAccuracy()

    def instance_metrics(self):
        metrics = []
        for index, class_name in enumerate(self.class_names):
            class_name_string = str(class_name)
            metrics = metrics + [Precision(class_id=index, name=class_name_string + "_precision", thresholds=0.5),
                                 Recall(class_id=index, name=class_name_string + "_recall", thresholds=0.5),
                                 RecallAtPrecision(0.99, class_id=index, name=class_name_string + "_r@p")]
        return metrics + [AUC(multi_label=True, num_labels=len(self.class_names), name="roc_auc"),
                          AUC(multi_label=True, curve="PR", num_labels=len(self.class_names), name="pr_auc")]

    def plot_metrics(self, steps_per_epoch):
        pass
