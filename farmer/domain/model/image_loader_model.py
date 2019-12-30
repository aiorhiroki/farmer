import os
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
    input_dir: str = None
    mask_dir: str = None
    data_list: str = None
    train_dirs: List[str] = field(default_factory=list)
    val_dirs: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    height: int = None
    width: int = None

    def get_task(self):
        if self.task == "segmentation":
            return Task.SEMANTIC_SEGMENTATION
        elif self.task == "classification":
            return Task.CLASSIFICATION
        else:
            raise NotImplementedError

    def get_class_names(self):
        if self.class_names:
            return self.class_names
        train_files = self._get_train_files()
        if self.task == ncc.tasks.Task.CLASSIFICATION:
            class_names = [
                os.path.basename(os.path.dirname(image_file))
                for image_file in train_files
            ]
            return sorted(list(set(class_names)))
        else:
            return ncc.readers.search_image_colors(train_files)

    def get_image_shape(self):
        if self.height and self.width:
            return self.height, self.width
        else:
            train_files = self._get_train_files()
            height, width, _ = ncc.readers.search_image_profile(train_files)
        return height, width

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
                            self.mask_dir,
                            "*" + image_ex
                        )
                    )
        return train_files
