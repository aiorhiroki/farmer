import numpy as np
import csv
from farmer import ncc


class PredictClassificationTask:
    def __init__(self, config):
        self.config = config

    def command(self, test_set, model, save_npy=True):
        prediction_gen = self._do_generate_batch_task(test_set)
        prediction = self._do_classification_predict_task(
            model, prediction_gen
        )
        self._do_save_result_task(test_set, prediction, save_npy)
        return prediction

    def _do_generate_batch_task(self, annotation_set):
        if self.config.framework == "tensorflow":
            test_gen = ncc.generators.ImageSequence(
                annotations=annotation_set,
                input_shape=(self.config.height, self.config.width),
                nb_classes=self.config.nb_classes,
                task=self.config.task,
                batch_size=4,
                input_data_type=self.config.input_data_type
            )

        elif self.config.framework == "pytorch":
            test_gen = ncc.generators.ImageDataset(
                annotations=annotation_set,
                input_shape=(self.config.height, self.config.width),
                nb_classes=self.config.nb_classes,
                task=self.config.task,
                batch_size=4,
                input_data_type=self.config.input_data_type
            )

        return test_gen

    def _do_classification_predict_task(
        self, model, annotation_gen
    ):
        if self.config.framework == "tensorflow":
            prediction = model.predict_generator(
                annotation_gen,
                steps=len(annotation_gen),
                workers=16 if self.config.multi_gpu else 1,
                max_queue_size=32 if self.config.multi_gpu else 10,
                use_multiprocessing=self.config.multi_gpu,
                verbose=1,
            )

        elif self.config.framework == "pytorch":
            # TODO: pytorch版classification
            raise NotImplementedError("[_do_classification_predict_task, pytorch] it is under construction.")

        return prediction

    def _do_save_result_task(self, annotation_set, prediction, save_npy):
        if save_npy:
            np.save(f"{self.config.info_path}/pred.npy", prediction)
        prediction_classes = np.argmax(prediction, axis=-1)
        pred_result = list()
        for files, prediction_cls in zip(annotation_set, prediction_classes):
            image_file, *_ = files
            pred_result.append(
                [
                    image_file,
                    self.config.class_names[int(prediction_cls)]
                ]
            )
        with open(f"{self.config.info_path}/pred.csv", "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(pred_result)
