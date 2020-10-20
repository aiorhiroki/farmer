import os
from farmer import ncc


class PredictSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, test_set, model):
        test_dataset = self._do_generate_batch_task(test_set)
        prediction = self._do_segmentation_predict_task(
            test_dataset, model, self.config.return_result
        )
        self._do_predict_on_video(model)
        return prediction

    def _do_generate_batch_task(self, test_set):
        sequence_args = dict(
            annotations=test_set,
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

    def _do_segmentation_predict_task(
        self, test_dataset, model, return_result=False
    ):

        # result_dir/image/test
        save_dir = os.path.join(self.config.image_path, "test")

        ncc.segmentation_metrics.generate_segmentation_result(
            nb_classes=self.config.nb_classes,
            dataset=test_dataset,
            model=model,
            save_dir=save_dir,
        )

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
