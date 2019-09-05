import os
import csv
import ncc


class ReadAnnotationTask:
    def __init__(self, config):
        self.config = config

    def command(self, phase):
        annotation_set = self._do_read_annotation_set_task(
            phase, self.config
        )
        self._do_write_annotations_task(
            phase, annotation_set,
            self.config.result_path, self.config.info_dir
        )

        return annotation_set

    def _do_read_annotation_set_task(
        self,
        phase: str,
        config
    ):

        target_dir = config.target_dir
        task = config.task
        class_names = config.class_names
        input_dir = config.input_dir
        mask_dir = config.mask_dir

        target_path = os.path.join(target_dir, phase)

        if task == ncc.tasks.Task.CLASSIFICATION:
            annotations = ncc.readers.classification_set(
                target_path, class_names
            )
        else:
            annotations = ncc.readers.segmentation_set(
                target_path, input_dir, mask_dir
            )

        return annotations

    def _do_write_annotations_task(
        self, phase, file_names, result_path, info_dir
    ):
        file_name = '{}_files.csv'.format(phase)
        csv_file_path = os.path.join(result_path, info_dir, file_name)

        with open(csv_file_path, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerows(file_names)
