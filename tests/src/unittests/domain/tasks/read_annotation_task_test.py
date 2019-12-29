import os
import shutil
import unittest
from farmer.domain.model.task_model import Task
from farmer.domain.model.trainer_model import Trainer
from farmer.domain.tasks.read_annotation_task import ReadAnnotationTask


class ReadAnnotationTaskTest(unittest.TestCase):
    def setUp(self):
        root_dir = "tests/test_result"
        result_dir = "20180905_1512"

        self.config = dict(
            epochs=100,
            batch_size=2,
            nb_classes=10,
            width=100,
            height=100,
            root_dir=root_dir,
            result_dir=result_dir,
        )

        os.makedirs(f"{root_dir}/{result_dir}/info")

    def tearDown(self):
        shutil.rmtree(self.config["root_dir"])

    def test_do_read_annotation_set_task_clssification(self):
        target_dir = "tests/data/classification"
        phase = "train"
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        expected = [
            [f"{target_dir}/{phase}/{class_name}/{image_id}.png", class_id]
            for class_id, class_name in enumerate(class_names)
            for image_id in range(10)
        ]

        self.config.update(
            dict(task=Task.CLASSIFICATION.value, target_dir=target_dir)
        )
        trainer = Trainer(**self.config)
        read_annotation_task = ReadAnnotationTask(trainer)
        actual = read_annotation_task._do_read_annotation_set_task(phase)

        self.assertEqual(sorted(actual), sorted(expected))

    def test_do_read_annotation_set_task_segmentation(self):
        target_dir = "tests/data/segmentation"
        input_dir = "image"
        mask_dir = "label"
        phase = "train"
        expected = [
            [
                f"{target_dir}/{phase}/{input_dir}/{image_id}.jpg",
                f"{target_dir}/{phase}/{mask_dir}/{image_id}.png",
            ]
            for image_id in range(10)
        ]

        self.config.update(
            dict(
                task=Task.SEMANTIC_SEGMENTATION.value,
                target_dir=target_dir,
                input_dir=input_dir,
                mask_dir=mask_dir,
            )
        )

        trainer = Trainer(**self.config)
        read_annotation_task = ReadAnnotationTask(trainer)
        actual = read_annotation_task._do_read_annotation_set_task(phase)

        self.assertEqual(sorted(actual), sorted(expected))

    def test_do_write_annotations_task(self):
        phase = "train"
        file_names = [
            ["image_1.png", "mask_1.png"],
            ["image_2.png", "mask_2.png"],
        ]

        trainer = Trainer(**self.config)
        read_annotation_task = ReadAnnotationTask(trainer)
        read_annotation_task._do_write_annotations_task(phase, file_names)
