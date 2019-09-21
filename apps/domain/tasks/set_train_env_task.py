import os
import random as rn
import multiprocessing as mp
import numpy as np

import tensorflow as tf
import torch


class SetTrainEnvTask:
    def __init__(self, config):
        self.config = config

    def command(self):
        self._do_set_random_seed_task()
        self._do_set_cpu_gpu_devices_task(self.config.gpu)
        self._do_create_dirs_task(result_path=self.config.result_path)

    def _do_set_random_seed_task(self, seed=1):
        # set random_seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        rn.seed(seed)
        if self.config.framework == "tensorflow":
            tf.random.set_seed(seed)
        elif self.config.framework == "pytorch":
            torch.manual_seed(0)
            # when running on CuDNN backend, two further options must be set:
            # warning: processing speed can be lower
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def _do_set_cpu_gpu_devices_task(self, gpu: str):
        # set gpu and cpu devices
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        num_threads = mp.cpu_count()
        if self.config.framework == "tensorflow":
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    def _do_create_dirs_task(self, result_path: str):
        # 結果を保存するディレクトリを目的別に作ります。
        log_dirs = ["image", "info", "learning", "model"]
        for log_dir in log_dirs:
            dir_path = os.path.join(result_path, log_dir)
            os.makedirs(dir_path, exist_ok=True)

        image_dirs = ["train", "validation", "test"]
        for image_dir in image_dirs:
            dir_path = os.path.join(result_path, "image", image_dir)
            os.makedirs(dir_path, exist_ok=True)
