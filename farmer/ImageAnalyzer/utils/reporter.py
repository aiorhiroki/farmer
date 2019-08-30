import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
import random as rn
import multiprocessing as mp
import numpy as np
import datetime
import os
from glob import glob
from configparser import ConfigParser
from farmer.ImageAnalyzer.task import Task
from farmer import app
import csv
from ncc.readers import search_image_profile
from ncc.utils import PostClient


class Reporter(Callback):
    ROOT_DIR = "result"
    IMAGE_DIR = "image"
    LEARNING_DIR = "learning"
    INFO_DIR = "info"
    MODEL_DIR = "model"
    PARAMETER = "parameter.txt"
    TRAIN_FILE = "train_files.csv"
    VALIDATION_FILE = "validation_files.csv"
    TEST_FILE = "test_files.csv"

    def __init__(self, config, shuffle=True, result_dir=None, training=True):
        super().__init__()
        self._create_dirs(result_dir)
        self._set_config_variables(config)
        self._set_env()
        self.train_files = self.read_annotation_set('train')
        self.validation_files = self.read_annotation_set('validation')
        self.test_files = self.read_annotation_set('test')
        self.height, self.width, self.channel = search_image_profile(
            [train_set[0] for train_set in self.train_files]
        )
        self._write_files(self.TRAIN_FILE, self.train_files)
        self._write_files(self.VALIDATION_FILE, self.validation_files)
        self.config['Data'] = dict(
            train_files=len(self.train_files),
            validation_files=len(self.validation_files)
        )
        self.save_params(self._parameter)
        self._init_milk()

    def _set_config_variables(self, config):
        self.secret_config = ConfigParser()
        self.secret_config.read('secret.ini')

    def _set_env(self):
        # set random_seed
        os.environ['PYTHONHASHSEED'] = '1'
        np.random.seed(1)
        rn.seed(1)
        tf.set_random_seed(1)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        core_num = mp.cpu_count()
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=core_num,
            inter_op_parallelism_threads=core_num
        )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

    def _write_files(self, csv_file, file_names):
        csv_path = os.path.join(self._info_dir, csv_file)
        with open(csv_path, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerows(file_names)

    def _create_dirs(self, result_dir):
        # 結果を保存するディレクトリを目的別に作ります。
        if result_dir is None:
            result_dir = datetime.datetime.today().strftime("%Y%m%d_%H%M")

        self._root_dir = self.ROOT_DIR
        self._result_dir = os.path.join(self._root_dir, result_dir)
        self._image_dir = os.path.join(self._result_dir, self.IMAGE_DIR)
        self.learning_dir = os.path.join(self._result_dir, self.LEARNING_DIR)
        self._info_dir = os.path.join(self._result_dir, self.INFO_DIR)
        self.model_dir = os.path.join(self._result_dir, self.MODEL_DIR)
        self._parameter = os.path.join(self._info_dir, self.PARAMETER)

        self.image_train_dir = os.path.join(
            self._image_dir, "train"
        )
        self.image_validation_dir = os.path.join(
            self._image_dir, "validation"
        )
        self.image_test_dir = os.path.join(
            self._image_dir, "test"
        )

        os.makedirs(self.image_train_dir, exist_ok=True)
        os.makedirs(self.image_validation_dir, exist_ok=True)
        os.makedirs(self.image_test_dir, exist_ok=True)
        os.makedirs(self.learning_dir, exist_ok=True)
        os.makedirs(self._info_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def save_params(self, filename):
        with open(filename, mode='w') as configfile:
            self.config.write(configfile)

    def _init_milk(self, training):
        if self.milk_id is None or not training:
            return
        self._milk_client = PostClient(
            root_url=app.config['MILK_API_URL']
        )
        self._milk_client.post(
            params=dict(
                train_id=int(self.milk_id),
                nb_classes=self.nb_classes,
                height=self.height,
                width=self.width,
                result_dir=os.path.abspath(self._result_dir),
                class_names=self.class_names
            ),
            route='first_config'
        )
        self._milk_client.close_session()

    def read_annotation_set(self, phase):
        target_dir_path = os.path.join(self.target_dir, phase)
        IMAGE_EXTENTINS = ['*.jpg', '*.png', '*.JPG']

        if self.task == Task.CLASSIFICATION:
            for class_id, class_name in enumerate(self.class_names):
                image_paths = list()
                class_dir_paths = [
                    os.path.join(target_dir_path,  class_name),
                    os.path.join(target_dir_path, '*',  class_name)
                ]
                for class_dir_path in class_dir_paths:
                    for image_ex in IMAGE_EXTENTINS:
                        image_paths += glob(class_dir_path, image_ex)

                annotations = [
                    [image_path, class_id] for image_path in image_paths
                ]

        elif self.task == Task.SEMANTIC_SEGMENTATION:
            mask_dir_paths = [
                os.path.join(target_dir_path, self.image_dir),
                os.path.join(target_dir_path, '*', self.image_dir)
            ]
            for mask_dir_path in mask_dir_paths:
                for image_ex in IMAGE_EXTENTINS:
                    mask_paths = glob(mask_dir_path, image_ex)

            annotations = [
                [
                    os.path.join(
                        os.path.splitext(mask_path)[0].replace(
                            '/{}/'.format(self.mask_dir),
                            '/{}/'.format(self.image_dir)
                        ),
                        image_ex
                    ),
                    mask_path
                ]
                for mask_path in mask_paths
                for image_ex in IMAGE_EXTENTINS
            ]

        else:
            raise NotImplementedError

        return annotations
