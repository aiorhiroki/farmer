import os
import csv
from farmer import ncc


class ReadAnnotationTask:
    def __init__(self, config):
        self.config = config

    def command(self, phase):
        annotation_set = self._do_read_annotation_set_task(phase)
        self._do_write_annotations_task(phase, annotation_set)

        return annotation_set

    def _do_read_annotation_set_task(self, phase: str):
        if phase == "train":
            data_list = self.config.train_dirs
        elif phase == "validation":
            data_list = self.config.val_dirs
        elif phase == "test":
            data_list = self.config.test_dirs

        print(f"{phase}: {data_list}")
        if self.config.task == ncc.tasks.Task.CLASSIFICATION:
            if self.config.input_data_type == "video":
                annotations = ncc.readers.classification_video_set(
                    self.config.target_dir,
                    data_list,
                    class_names=self.config.class_names,
                    csv_file=self.config.video_csv,
                    skip_frame=self.config.skip_frame,
                    time_format=self.config.time_format
                )
            else:
                annotations = ncc.readers.classification_set(
                    self.config.target_dir, data_list, self.config.class_names
                )
        elif self.config.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
            annotations = ncc.readers.segmentation_set(
                self.config.target_dir,
                data_list,
                self.config.input_dir,
                self.config.label_dir
            )
        elif self.config.task == ncc.tasks.Task.OBJECT_DETECTION:
            ncc.readers.detection_set(
                self.config.target_dir,
                data_list,
                self.config.input_dir,
                self.config.label_dir,
                csv_file=f"{self.config.info_path}/{phase}.csv",
                class_names=self.config.class_names
            )
            annotations = [[f"annotations saved in {phase}.csv"]]

        if phase == "train":
            self.config.nb_train_data = len(annotations)
        elif phase == "validation":
            self.config.nb_validation_data = len(annotations)
        elif phase == "test":
            self.config.nb_test_data = len(annotations)

        return annotations

    def _do_write_annotations_task(self, phase: str, annotations: list):
        file_name = "{}_files.csv".format(phase)
        csv_file_path = os.path.join(self.config.info_path, file_name)

        with open(csv_file_path, "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(annotations)
