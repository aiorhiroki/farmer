import ncc
import numpy as np
import sklearn


class EvalClassificationTask:

    def __init__(self, config):
        self.config = config

    def command(self, prediction, annotation_set):
        eval_report = self._do_classification_evaluation_task(
            prediction, annotation_set
        )
        return eval_report

    def _do_classification_evaluation_task(
        self,
        prediction,
        annotation_set
    ):
        prediction_cls = np.argmax(prediction, axis=1)
        true_cls = [class_id for _, class_id in annotation_set]
        true = np.eye(self.config.nb_classes, dtype=np.uint8)[true_cls]
        eval_report = sklearn.metrics.classification_report(
            true_cls, prediction_cls, output_dict=True
        )
        fpr, tpr, auc = ncc.metrics.roc(
            true, prediction, self.config.nb_classes, show_plot=False
        )
        eval_report.update(
            dict(
                fpr=list(fpr['macro']),
                tpr=list(tpr['macro']),
                auc=auc['macro']
            )
        )
        return eval_report
