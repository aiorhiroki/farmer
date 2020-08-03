import os
import shutil
import random as rn
import multiprocessing as mp
import numpy as np

import tensorflow as tf


class SetTrainEnvTask:
    def __init__(self, config):
        self.config = config

    def command(self):
        self._do_set_random_seed_task()
        self._do_set_cpu_gpu_devices_task(self.config.gpu)
        self._do_create_dirs_task()

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
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                for k in range(len(physical_devices)):
                    tf.config.experimental.set_memory_growth(physical_devices[k], True)
                    print(f'memory growth: ', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        num_threads = mp.cpu_count()
        if self.config.framework == "tensorflow":
            tf.config.threading.set_inter_op_parallelism_threads(num_threads)
            tf.config.threading.set_intra_op_parallelism_threads(num_threads)

    def _do_create_dirs_task(self):
        # 結果を保存するディレクトリを目的別に作ります。
        if self.config.trial_number is None or self.config.trial_number == 0:
            # infoはtrial共通
            dir_path = self.config.info_path
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        log_dirs = [self.config.model_path, self.config.learning_path, self.config.image_path]
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
