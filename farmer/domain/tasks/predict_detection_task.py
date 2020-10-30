from farmer import ncc
import numpy as np


class PredictDetectionTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model):
        eval_report = self._do_evaluation_task(model)
        return eval_report

    def _do_evaluation_task(self, model):
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
