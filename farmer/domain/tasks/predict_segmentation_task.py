from farmer import ncc
import os
import cv2
import numpy as np
import tqdm


class PredictSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, test_set, model):
        prediction = self._do_segmentation_predict_task(
            test_set, model, self.config.return_result
        )
        return prediction

    def _do_segmentation_predict_task(
        self, test_set, model, return_result=False
    ):
        save_dir = f"{self.config.image_path}/test"

        if self.config.framework == "tensorflow":
            ncc.utils.generate_segmentation_result(
                nb_classes=self.config.nb_classes,
                height=self.config.height,
                width=self.config.width,
                annotations=test_set,
                model=model,
                save_dir=save_dir,
                framework=self.config.framework,
                train_colors=self.config.train_colors,
            )

        elif self.config.framework == "pytorch":
            print('[_do_segmentation_predict_task, pytorch] it is under construction')
