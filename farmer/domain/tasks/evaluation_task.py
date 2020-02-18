from farmer import ncc
import numpy as np
from sklearn.metrics import classification_report


class EvaluationTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model=None, prediction=None):
        eval_report = self._do_evaluation_task(
            annotation_set, model, prediction
        )
        return eval_report

    def _do_evaluation_task(self, annotation_set, model, prediction):

        if self.config.framework == 'tensorflow':
            if self.config.task == ncc.tasks.Task.CLASSIFICATION:
                prediction_cls = np.argmax(prediction, axis=1)
                true_cls = [class_id for *_, class_id in annotation_set]
                eval_report = classification_report(
                    true_cls, prediction_cls, output_dict=True
                )
            elif self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                eval_report = ncc.metrics.iou_dice_val(
                    self.config.nb_classes,
                    self.config.height,
                    self.config.width,
                    annotation_set,
                    model,
                    self.config.train_colors
                )
            elif self.config.task == ncc.tasks.Task.OBJECT_DETECTION:
                from keras_retinanet.bin import evaluate
                classes = f"{self.config.info_path}/classes.csv"
                test_annotations = f"{self.config.info_path}/test.csv"
                evaluate.main(
                    [
                        "csv", test_annotations, classes,
                        model,
                        "--convert-model"
                    ]
                )
                eval_report = ["evaluation result is std out"]

        elif self.config.framework == 'pytorch':
            print('[EvaluationTask, _do_evaluation_task, pytorch] under construction')

        return eval_report
