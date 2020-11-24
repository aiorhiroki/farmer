import numpy as np
import csv
from farmer import ncc
from sklearn.metrics import classification_report


class PredictClassificationTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model, save_npy=True):
        dataset, generator = self._do_generate_batch_task(annotation_set)
        prediction, eval_report = self._do_classification_predict_task(
            model, dataset, generator
        )
        self._do_predict_on_video(model)
        self._do_save_result_task(annotation_set, prediction, save_npy)
        return eval_report

    def _do_generate_batch_task(self, annotation_set):
        dataset = ncc.generators.ClassificationDataset(
            annotations=annotation_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            augmentation=[],
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )
        generator = ncc.generators.Dataloder(dataset, batch_size=1, shuffle=False)
        return dataset, generator

    def _do_classification_predict_task(self, model, dataset, generator):
        prediction = model.predict(
            generator,
            steps=len(generator),
            workers=16 if self.config.multi_gpu else 1,
            max_queue_size=32 if self.config.multi_gpu else 10,
            use_multiprocessing=self.config.multi_gpu,
            verbose=1,
        )
        prediction_cls = np.argmax(prediction, axis=1)
        true_cls = np.argmax(
            [class_id for _, class_id in dataset], axis=1)
        eval_report = classification_report(
            true_cls, prediction_cls, output_dict=True
        )
        return prediction, eval_report

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
        ncc.metrics.show_matrix(
            true_ids, pred_ids, self.config.class_names, self.config.info_path)

    def _do_predict_on_video(self, model):
        if len(self.config.predict_videos) == 0:
            return
        classifier = ncc.predictions.Classifier(
            self.config.class_names, model)
        for predict_video in self.config.predict_videos:
            video_path = predict_video["name"]
            start = predict_video.get("start_time")
            end = predict_video.get("end_time")

            ncc.video.predict_on_video(
                classifier, video_path, self.config.video_path, start, end)
