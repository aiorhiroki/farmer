import dataclasses
import os
from datetime import datetime
from .config_model import Config
from .train_params_model import TrainParams
from .image_loader_model import ImageLoader


@dataclasses.dataclass
class Trainer(Config, ImageLoader):
    train_id: int = None
    training: bool = None
    epochs: int = None
    steps: int = None
    gpu: str = None
    nb_gpu: int = None
    multi_gpu: bool = None
    trained_path: str = None
    trained_model_path: str = None
    nb_train_data: int = 0
    nb_validation_data: int = 0
    nb_test_data: int = 0
    save_pred: bool = True
    segmentation_val_step: int = 3
    n_splits: int = 5
    batch_period: int = 100
    early_stopping: bool = False
    patience: int = 10
    monitor: str = "val_loss"
    optuna: bool = False
    seed: int = 1
    n_trials: int = 10
    timeout: int = None
    trial_number: int = None
    trial_params: dict = None
    train_params: TrainParams = None

    def __post_init__(self):
        self.task = self.get_task()
        self.gpu = str(self.gpu)
        self.nb_gpu = len(self.gpu.split(",")) if self.gpu else 0
        self.multi_gpu = self.nb_gpu > 1
        self.train_params.batch_size *= self.nb_gpu if self.multi_gpu else 1
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
        if not self.train_params.class_weights:
            self.train_params.class_weights = {
                class_id: 1.0 for class_id in range(self.nb_classes)
            }

        # For optuna analysis hyperparameter
        def _check_need_optuna(train_params: dict):
            for val in train_params.values():
                if isinstance(val, list):
                    self.optuna = True
                elif isinstance(val, dict):
                    _check_need_optuna(val)

        _check_need_optuna(dataclasses.asdict(self.train_params))
