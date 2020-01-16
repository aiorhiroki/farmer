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
        ncc.utils.generate_segmentation_result(
            nb_classes=self.config.nb_classes,
            height=self.config.height,
            width=self.config.width,
            annotations=test_set,
            model=model,
            save_dir=save_dir,
            train_colors=self.config.train_colors
        )
