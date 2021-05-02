import os
import shutil
import random as rn
import multiprocessing as mp
import numpy as np
import copy

import tensorflow as tf
from farmer.domain.model import TrainParams


class SetTrainEnvTask:
    def __init__(self, config):
        self.config = config

    def command(self, trial):
        self._do_set_random_seed_task()
        self._do_set_cpu_gpu_devices_task(self.config.gpu)
        self._do_set_train_params_task(trial)
        self._do_create_dirs_task()

        return self.config

    def _do_set_random_seed_task(self):
        seed = self.config.seed
        # set random_seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        rn.seed(seed)
        if self.config.framework == "tensorflow":
            tf.random.set_seed(seed)
            # tf.set_random_seed(seed)

    def _do_set_cpu_gpu_devices_task(self, gpu: str):
        # set gpu and cpu devices
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
            # GPUメモリ使用量を抑える
            devices = tf.config.experimental.list_physical_devices('GPU')
            if len(devices) > 0:
                for k in range(len(devices)):
                    tf.config.experimental.set_memory_growth(devices[k], True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        num_threads = mp.cpu_count()
        if self.config.framework == "tensorflow":
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    def _do_set_train_params_task(self, trial):
        def set_train_params(train_params: dict) -> dict:
            for key, val in train_params.items():
                if isinstance(val, dict):
                    set_train_params(val)
                elif isinstance(val, list):
                    if len(val) == 0:
                        continue
                    if isinstance(val[0], str):
                        train_params[key] = trial.suggest_categorical(key, val)
                    elif isinstance(val[0], (int, float)):
                        if len(val) == 2:
                            train_params[key] = trial.suggest_loguniform(
                                key, *val)
                        elif len(val) == 3:
                            param_val = trial.suggest_discrete_uniform(
                                key, *val
                            )
                            if int(param_val) == param_val:
                                param_val = int(param_val)
                            train_params[key] = param_val
                    elif isinstance(val[0], dict):
                        train_params[key] = dict(
                            name=trial.suggest_categorical(
                                f"{key}_name", [v_dic["name"] for v_dic in val]
                            )
                        )
                        for val_dic in val:
                            if val_dic["name"] == train_params[key]["name"]:
                                functions = val_dic["functions"]
                                train_params[key]["functions"] = functions
                                set_train_params(functions)

        if self.config.optuna:
            self.config.trial_number = trial.number
            self.config.trial_params = trial.params
            # result_dir/trial#/learning/
            trial_result_dir = f'{self.config.result_path}/trial{trial.number}'
            self.config.learning_path = os.path.join(
                trial_result_dir, self.config.learning_dir)
            self.config.model_path = os.path.join(
                trial_result_dir, self.config.model_dir)
            self.config.image_path = os.path.join(
                trial_result_dir, self.config.image_dir)
            self.config.tfboard_path = os.path.join(
                trial_result_dir, self.config.tfboard_dir)

            # set train params to params setted by optuna
            train_params_dict = copy.deepcopy(self.config.optuna_params)
            set_train_params(train_params_dict)

        else:
            train_params_dict = self.config.train_params

        self.config.train_params = TrainParams(**train_params_dict)

    def _do_create_dirs_task(self):
        # 結果を保存するディレクトリを目的別に作ります。
        if self.config.trial_number is None or self.config.trial_number == 0:
            # infoはtrial共通
            dir_path = self.config.info_path
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        log_dirs = [
            self.config.model_path,
            self.config.learning_path,
            self.config.image_path,
            self.config.video_path,
            self.config.tfboard_path
        ]
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir)

        image_dirs = ["train", "validation", "test"]
        for image_dir in image_dirs:
            dir_path = os.path.join(self.config.image_path, image_dir)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
