from farmer import ncc
import os
import cv2
import numpy as np
import tqdm


class PredictSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, model, test_set):
        prediction = self._do_segmentation_predict_task(
            model, test_set, self.config.return_result
        )
        return prediction

    def _do_segmentation_predict_task(
        self, model, test_set, return_result=False
    ):
        image_util = ncc.utils.ImageUtil(
            self.config.nb_classes, (self.config.height, self.config.width)
        )
        result = list()
        for input_file, _ in tqdm.tqdm(test_set):
            file_name = os.path.basename(input_file)
            input_image = image_util.read_image(input_file, anti_alias=True)
            prediction = model.predict(np.expand_dims(input_image, axis=0))
            output = image_util.blend_image(
                prediction[0], image_util.current_raw_size
            )
            cv2.imwrite(
                os.path.join(self.config.image_test_dir, file_name), output
            )
            if return_result:
                result.append(output)

        if return_result:
            return result
