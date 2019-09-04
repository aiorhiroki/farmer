from dataclasses import dataclass
import os
from datetime import datetime
from .config_model import Config
from .image_loader_model import ImageLoader


@dataclass
class Trainer(Config, ImageLoader):
    train_id: int = None
    epochs: int = None
    batch_size: int = None
    learning_rate: float = None
    optimizer: str = None
    augmentation: bool = None
    gpu: str = None
    nb_gpu: int = None
    multi_gpu: bool = None
    loss: str = None
    trained_model_path: str = None
    model_name: str = None
    backbone: str = None
    nb_train_data: int = 0
    nb_validation_data: int = 0

    def __post_init__(self):
        self.train_id = self.getint(self.train_id)
        self.epochs = self.getint(self.epochs)
        self.batch_size = self.getint(self.batch_size)
        self.learning_rate = self.getfloat(self.learning_rate)
        self.augmentation = self.getboolean(self.augmentation)
        self.task = self.getint(self.task)

        self.nb_gpu = len(self.gpu.split(',')) if self.gpu else 0
        self.multi_gpu = self.nb_gpu > 1
        self.batch_size *= self.nb_gpu if self.multi_gpu else 1

        if self.result_dir is None:
            self.result_dir = datetime.today().strftime("%Y%m%d_%H%M")
        self.result_path = os.path.join(self.root_dir, self.result_dir)
        self.info_path = os.path.join(self.result_path, self.info_dir)
        self.model_path = os.path.join(self.result_path, self.model_dir)
        self.learning_path = os.path.join(self.result_path, self.learning_dir)
        self.image_path = os.path.join(self.result_path, self.image_dir)

        self.class_names = self.get_class_names()
        self.nb_classes = self.getint(self.nb_classes)
        self.height, self.width = self.get_image_shape()
