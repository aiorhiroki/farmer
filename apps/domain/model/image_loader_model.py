import os
import json
from glob import glob
from dataclasses import dataclass, field
from typing import List
import ncc


@dataclass
class ImageLoader:
    task: int = None
    target_dir: str = None
    class_names: List[str] = field(default_factory=list)
    nb_classes: int = None
    input_dir: str = None
    mask_dir: str = None
    data_list: str = None
    train_dirs: List[str] = field(default_factory=list)
    val_dirs: List[str] = field(default_factory=list)
    test_dirs: List[str] = field(default_factory=list)
    height: int = None
    width: int = None

    def get_data_list(self):
        with open(self.data_list, "r") as filereader:
            data_list = json.load(filereader)
        return data_list["train"], data_list["validation"], data_list["test"]

    def get_class_names(self):
        if self.class_names:
            return self.class_names.split()
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
            height = self.getint(self.height)
            width = self.getint(self.width)
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
            return str_bool == "True" or str_bool == "yes"
