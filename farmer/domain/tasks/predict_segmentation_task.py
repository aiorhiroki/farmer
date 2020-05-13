import os
from farmer import ncc


class PredictSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, test_set, model, trial=None):
        prediction = self._do_segmentation_predict_task(
            test_set, model, self.config.return_result, trial
        )
        return prediction

    def _do_segmentation_predict_task(
        self, test_set, model, return_result=False, trial=None
    ):
        if trial:
            # result_dir/trial#/image/validation/
            trial_image_path = self.config.image_path.split('/')
            trial_image_path.insert(-1, f"trial{trial.number}")
            if trial_image_path[0] == '':
                trial_image_path[0] = '/'
            save_dir = os.path.join(*trial_image_path, "test")
        else:
            save_dir = os.path.join(self.config.image_path, "test")

        ncc.segmentation_metrics.generate_segmentation_result(
            nb_classes=self.config.nb_classes,
            height=self.config.height,
            width=self.config.width,
            annotations=test_set,
            model=model,
            save_dir=save_dir,
            train_colors=self.config.train_colors
        )
