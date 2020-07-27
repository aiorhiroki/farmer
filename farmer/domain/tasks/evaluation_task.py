from farmer import ncc
import numpy as np
from sklearn.metrics import classification_report


class EvaluationTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model=None, prediction=None):
        annotation_dataset = self._do_generate_batch_task(annotation_set)
        eval_report = self._do_evaluation_task(
            annotation_dataset, model, prediction
        )
        return eval_report

    def _do_generate_batch_task(self, annotation_set):
        sequence_args = dict(
            annotations=annotation_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            augmentation=[],
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )

        if self.config.task == ncc.tasks.Task.CLASSIFICATION:
            return ncc.generators.ClassificationDataset(**sequence_args)

        elif self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            return ncc.generators.SegmentationDataset(**sequence_args)

    def _do_evaluation_task(self, annotation_dataset, model, prediction):
        if self.config.task == ncc.tasks.Task.CLASSIFICATION:
            prediction_cls = np.argmax(prediction, axis=1)
            true_cls = np.argmax([class_id for _, class_id in annotation_dataset], axis=1)
            eval_report = classification_report(
                true_cls, prediction_cls, output_dict=True
            )
        elif self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            eval_report = ncc.metrics.iou_dice_val(
                self.config.nb_classes,
                annotation_dataset,
                model,
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
        return eval_report
