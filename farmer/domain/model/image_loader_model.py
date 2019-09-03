import os
from glob import glob
from dataclasses import dataclass, field
from typing import List
import ncc


@dataclass
class ImageLoader:
    task: int = None
    target_dir: str
    class_names: List[str] = field(default_factory=list)
    nb_classes: int = None
    image_dir: str = None
    mask_dir: str = None
    height: int = None
    width: int = None

    def __post_init__(self):
        self.class_names = self.get_class_names()
        self.nb_classes = self.getint(self.nb_classes)
        self.height, self.width = self.get_image_shape()

    def get_class_names(self):
        if self.class_names:
            return self.class_names
        train_files = self._get_train_files()
        if self.task == ncc.tasks.Task.CLASSIFICATION:
            class_names = [
                os.path.basename(
                    os.path.dirname(image_file)
                )
                for image_file in train_files
            ]
            return list(set(class_names))
        else:
            return ncc.readers.search_image_colors(train_files)

    def get_image_shape(self):
        if self.height and self.width:
            height = self.getint(self.height)
            width = self.getint(self.width)
        else:
            train_files = self._get_train_files()
            height, width, _ = ncc.readers.search_image_profile(train_files)
        return height, width

    def _get_train_files(self):
        IMAGE_EXTENTINS = ['.jpg', '.png', '.JPG']
        train_files = list()
        for image_ex in IMAGE_EXTENTINS:
            train_files += glob(
                self.target_dir, 'train', '*', image_ex
            )
        return train_files

    @classmethod
    def getint(cls, str_number):
        if str_number:
            return int(str_number)

    @classmethod
    def getfloat(cls, str_number):
        if str_number:
            return float(str_number)

    @classmethod
    def getboolean(cls, str_bool):
        if str_bool:
            return str_bool == 'True' or str_bool == 'yes'
