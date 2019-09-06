import os
import random as rn
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from keras import backend as K


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
        tf.set_random_seed(seed)

    def _do_set_cpu_gpu_devices_task(self, gpu: str):
        # set gpu and cpu devices
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        core_num = mp.cpu_count()
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=core_num,
            inter_op_parallelism_threads=core_num,
        )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

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
