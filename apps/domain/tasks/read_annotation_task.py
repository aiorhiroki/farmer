import os
import csv
import ncc


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

        if self.config.task == ncc.tasks.Task.CLASSIFICATION:
            annotations = ncc.readers.classification_set(
                self.config.target_dir, data_list, self.config.class_names
            )
        else:
            annotations = ncc.readers.segmentation_set(
                self.config.target_dir,
                data_list,
                self.config.input_dir,
                self.config.mask_dir
            )
        return annotations

    def _do_write_annotations_task(self, phase: str, file_names: list):
        file_name = "{}_files.csv".format(phase)
        csv_file_path = os.path.join(self.config.info_path, file_name)

        with open(csv_file_path, "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(file_names)
