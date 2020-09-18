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
        dataset = ncc.generators.ClassificationDataset(
            annotations=annotation_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            augmentation=[],
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )
        return ncc.generators.Dataloder(dataset, batch_size=1, shuffle=False)

    def _do_classification_predict_task(
        self, model, annotation_gen
    ):
        prediction = model.predict_generator(
            annotation_gen,
            steps=len(annotation_gen),
            workers=16 if self.config.multi_gpu else 1,
            max_queue_size=32 if self.config.multi_gpu else 10,
            use_multiprocessing=self.config.multi_gpu,
            verbose=1,
        )
        return prediction

    def _do_save_result_task(self, annotation_set, prediction, save_npy):
        if save_npy:
            np.save(f"{self.config.info_path}/pred.npy", prediction)

        prediction_classes = np.argmax(prediction, axis=-1)
        pred_result = list()
        pred_ids = list()
        true_ids = list()
        for files, pred_cls in zip(annotation_set, prediction_classes):
            if self.config.input_data_type == "video":
                image_file, frame_id, true_cls = files
                pred_result.append(
                    [
                        image_file,
                        frame_id,
                        self.config.class_names[int(pred_cls)],
                        self.config.class_names[int(true_cls)]
                    ]
                )
            else:
                image_file, true_cls = files
                pred_result.append(
                    [
                        image_file,
                        self.config.class_names[int(pred_cls)],
                        self.config.class_names[int(true_cls)]
                    ]
                )
            pred_ids.append(int(pred_cls))
            true_ids.append(int(true_cls))
        with open(f"{self.config.info_path}/pred.csv", "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(pred_result)
        ncc.metrics.show_matrix(true_ids, pred_ids, self.config.class_names)
