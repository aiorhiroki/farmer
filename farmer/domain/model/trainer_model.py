from dataclasses import dataclass
import copy
import os
from datetime import datetime
from dataclasses import field
from typing import List, Dict
from .config_model import Config
from .image_loader_model import ImageLoader


@dataclass
class Trainer(Config, ImageLoader):
    train_id: int = None
    training: bool = None
    epochs: int = None
    steps: int = None
    batch_size: int = None
    learning_rate: float = None
    optimizer: str = None
    augmentation: List[str] = field(default_factory=list)
    gpu: str = None
    nb_gpu: int = None
    multi_gpu: bool = None
    loss: str = None
    loss_params: Dict[str, float] = field(default_factory=dict)
    trained_path: str = None
    trained_model_path: str = None
    model_name: str = None
    backbone: str = None
    activation: str = "softmax"
    nb_train_data: int = 0
    nb_validation_data: int = 0
    nb_test_data: int = 0
    save_pred: bool = True
    segmentation_val_step: int = 3
    n_splits: int = 5
    batch_period: int = 100
    cosine_decay: bool = False
    cosine_lr_max: int = 0.01
    cosine_lr_min: int = 0.001
    optuna: bool = False
    seed: int = 1
    op_batch_size: int = None
    op_learning_rate: float = None
    op_optimizer: str = None
    op_backbone: str = None
    op_loss: str = None
    n_trials: int = 3
    timeout: int = 3 * 60 * 60
    trial_number: int = None
    trial_params: dict = None

    def __post_init__(self):
        self.task = self.get_task()
        self.gpu = str(self.gpu)
        self.nb_gpu = len(self.gpu.split(",")) if self.gpu else 0
        self.multi_gpu = self.nb_gpu > 1
        if self.batch_size:
            self.batch_size *= self.nb_gpu if self.multi_gpu else 1
        if self.result_dir is None:
            self.result_dir = datetime.today().strftime("%Y%m%d_%H%M")
        self.target_dir = os.path.join(self.root_dir, self.target_dir)
        if self.trained_path is not None:
            self.trained_path = os.path.join(self.root_dir, self.trained_path)
            if self.trained_path.endswith('.h5'):
                self.trained_model_path = self.trained_path
            else:
                self.trained_model_path = os.path.join(
                    self.trained_path, "model/last_model.h5"
                )
        self.result_path = os.path.join(
            self.root_dir, self.result_root_dir, self.result_dir)
        self.info_path = os.path.join(self.result_path, self.info_dir)
        self.model_path = os.path.join(self.result_path, self.model_dir)
        self.learning_path = os.path.join(self.result_path, self.learning_dir)
        self.image_path = os.path.join(self.result_path, self.image_dir)
        self.get_train_dirs()
        self.train_dirs = [str(train_dir) for train_dir in self.train_dirs]
        self.val_dirs = [str(val_dir) for val_dir in self.val_dirs if val_dir]
        self.test_dirs = [str(test_dir) for test_dir in self.test_dirs]
        self.class_names = self.get_class_names()
        self.nb_classes = len(self.class_names)
        self.height, self.width = self.get_image_shape()
        self.mean, self.std = None, None

        # For optuna analysis hyperparameter
        batch_size_is_list = type(self.batch_size) == list
        learning_rate_is_list = type(self.learning_rate) == list
        optimizer_is_list = type(self.optimizer) == list
        backbone_is_list = type(self.backbone) == list
        loss_is_list = type(self.loss) == list
        
        if batch_size_is_list:
            self.op_batch_size = copy.deepcopy(self.batch_size)
        if learning_rate_is_list:
            self.op_learning_rate = copy.deepcopy(self.learning_rate)
        if optimizer_is_list:
            self.op_optimizer = copy.deepcopy(self.optimizer)
        if backbone_is_list:
            self.op_backbone = copy.deepcopy(self.backbone)
        if loss_is_list:
            self.op_loss = copy.deepcopy(self.loss)

        self.optuna = any((
            batch_size_is_list,
            learning_rate_is_list,
            optimizer_is_list,
            backbone_is_list,
            loss_is_list
        ))

