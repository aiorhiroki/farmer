import os
from farmer import ncc


class PredictSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model):
        dataset = self._do_generate_batch_task(annotation_set)
        eval_report = self._do_segmentation_predict_task(dataset, model)
        self._do_predict_on_video(model)
        return eval_report

    def _do_generate_batch_task(self, annotation_set):
        dataset = ncc.generators.SegmentationDataset(
            annotations=annotation_set,
            input_shape=(self.config.height, self.config.width),
            nb_classes=self.config.nb_classes,
            augmentation=[],
            train_colors=self.config.train_colors,
            input_data_type=self.config.input_data_type
        )
        return dataset

    def _do_segmentation_predict_task(self, dataset, model):
        # result_dir/image/test
        save_dir = os.path.join(self.config.image_path, "test")

        eval_report = ncc.segmentation_metrics.generate_segmentation_result(
            nb_classes=self.config.nb_classes,
            dataset=dataset,
            model=model,
            save_dir=save_dir,
            batch_size=self.config.train_params.batch_size,
            sdice_tolerance=self.config.sdice_tolerance
        )
        return eval_report

    def _do_predict_on_video(self, model):
        if len(self.config.predict_videos) == 0:
            return
        segmenter = ncc.predictions.Segmenter(model)
        for predict_video in self.config.predict_videos:
            video_path = predict_video["name"]
            start = predict_video.get("start_time")
            end = predict_video.get("end_time")

            ncc.video.predict_on_video(
                segmenter, video_path, self.config.video_path, start, end)
