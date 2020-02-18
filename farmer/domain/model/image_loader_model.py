import os
import csv
from glob import glob
from dataclasses import dataclass, field
from typing import List
from farmer import ncc
from farmer.domain.model.task_model import Task


@dataclass
class ImageLoader:
    task: int = None
    target_dir: str = None
    class_names: List[str] = field(default_factory=list)
    train_colors: List[int] = field(default_factory=list)
    input_dir: str = None
    label_dir: str = None
    train_dirs: List[str] = field(default_factory=list)
    val_dirs: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    height: int = None
    width: int = None
    input_data_type: str = "image"
    skip_frame: int = 30
    time_format: str = "datetime"

    def get_task(self):
        if self.task == "segmentation":
            return Task.SEMANTIC_SEGMENTATION
        elif self.task == "classification":
            return Task.CLASSIFICATION
        elif self.task == "detection":
            return Task.OBJECT_DETECTION
        else:
            raise NotImplementedError

    def get_class_names(self):
        if self.class_names:
            return [str(class_name) for class_name in self.class_names]
        if not self.training and self.trained_path:
            class_name_files = f"{self.trained_path}/info/classes.csv"
            class_names = list()
            class_ids = list()
            with open(class_name_files, "r") as fr:
                reader = csv.reader(fr)
                for class_name, class_id in reader:
                    class_names.append(class_name)
                    class_ids.append(class_id)
            if self.task == ncc.tasks.Task.SEMANTIC_SEGMENTATION:
                self.train_colors = class_ids
            return class_names
        train_files = self._get_train_files()
        if self.task == ncc.tasks.Task.CLASSIFICATION:
            class_names = [
                os.path.basename(os.path.dirname(image_file))
                for image_file in train_files
            ]
            return sorted(list(set(class_names)))
        else:
            self.train_colors = ncc.readers.search_image_colors(train_files)
            return self.train_colors

    def get_image_shape(self):
        if self.height and self.width:
            return self.height, self.width
        else:
            train_files = self._get_train_files()
            height, width, _ = ncc.readers.search_image_profile(train_files)
        return height, width

    def get_train_dirs(self):
        if self.training and (self.train_dirs is None or len(self.train_dirs) == 0):
            self.train_dirs = [
                d for d in os.listdir(self.target_dir)
                if os.path.isdir(f"{self.target_dir}/{d}")
            ]

    def _get_train_files(self):
        IMAGE_EXTENTINS = [".jpg", ".png"]
        train_files = list()
        for image_ex in IMAGE_EXTENTINS:
            for train_dir in self.train_dirs:
                if self.task == ncc.tasks.Task.CLASSIFICATION:
                    train_files += glob(
                        os.path.join(
                            self.target_dir, train_dir, "*", "*" + image_ex
                        )
                    )
                else:
                    train_files += glob(
                        os.path.join(
                            self.target_dir,
                            train_dir,
                            self.label_dir,
                            "*" + image_ex
                        )
                    )
        return train_files
