import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
import random as rn
import multiprocessing as mp
from .model import build_model
from .milk_client import MilkClient
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob
from configparser import ConfigParser
from farmer.ImageAnalyzer.task import Task
import csv

from ncc.readers import search_image_profile
from ncc.utils import palette, MatPlot, slack_logging
from ncc.utils import get_imageset, ImageUtil


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
        self._create_plot_figures()
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
        self.image_util = ImageUtil(self.nb_classes, (self.height, self.width))
        self._init_milk()

    def _set_config_variables(self, config):
        # configに入っている値をインスタンス変数にする。
        self.config = config
        config_params = self.config['project_settings']
        self.epoch = config_params.getint('epoch')
        self.batch_size = config_params.getint('batch_size')
        self.learning_rate = config_params.getfloat('learning_rate')
        self.optimizer = config_params.get('optimizer')
        self.augmentation = config_params.get('augmentation') == 'yes'
        self.gpu = config_params.get('gpu') or '-1'
        self.loss = config_params.get('loss')
        self.model_path = config_params.get('model_path')
        self.target_dir = config_params.get('target_dir')
        self.class_names = config_params.get('class_names')
        self.nb_classes = len(self.class_names)
        self.image_dir = config_params.get('image_folder')
        self.mask_dir = config_params.get('mask_folder')

        self.nb_gpu = len(self.gpu.split(',')) if self.gpu else 0
        self.multi_gpu = self.nb_gpu > 1
        self.batch_size *= self.nb_gpu if self.multi_gpu else 1

        self.model_name = config_params.get('model_name')
        self.height = config_params.getint('height')
        self.width = config_params.getint('width')
        self.backbone = config_params.get('backbone')

        self.task = config_params.getint('task_id')
        self.milk_id = config_params.get('id')

        if self.task == Task.CLASSIFICATION:
            self.metric = 'acc'
        elif self.task == Task.SEMANTIC_SEGMENTATION:
            self.metric = 'iou_score'
        else:
            raise NotImplementedError

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
        self._learning_dir = os.path.join(self._result_dir, self.LEARNING_DIR)
        self._info_dir = os.path.join(self._result_dir, self.INFO_DIR)
        self.model_dir = os.path.join(self._result_dir, self.MODEL_DIR)
        self._parameter = os.path.join(self._info_dir, self.PARAMETER)

        self._image_train_dir = os.path.join(
            self._image_dir, "train"
        )
        self._image_validation_dir = os.path.join(
            self._image_dir, "validation"
        )
        self.image_test_dir = os.path.join(
            self._image_dir, "test"
        )

        os.makedirs(self._image_train_dir, exist_ok=True)
        os.makedirs(self._image_validation_dir, exist_ok=True)
        os.makedirs(self.image_test_dir, exist_ok=True)
        os.makedirs(self._learning_dir, exist_ok=True)
        os.makedirs(self._info_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def save_params(self, filename):
        with open(filename, mode='w') as configfile:
            self.config.write(configfile)

    def _init_milk(self, training):
        if self.milk_id is None or not training:
            return
        self._milk_client = MilkClient()
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

    def save_image_from_ndarray(self, train_set, validation_set,
                                palette, epoch, index_void=None):
        assert len(train_set) == len(validation_set) == 3
        train_image = get_imageset(
            image_in_np=train_set[0],
            image_out_np=train_set[1],
            image_gt_np=train_set[2],
            palette=palette,
            index_void=index_void
        )
        validation_image = get_imageset(
            image_in_np=validation_set[0],
            image_out_np=validation_set[1],
            image_gt_np=validation_set[2],
            palette=palette,
            index_void=index_void
        )
        file_name = 'epoch_{}.png'.format(epoch)
        train_filename = os.path.join(
            self._image_train_dir, file_name
        )
        validation_filename = os.path.join(
            self._image_validation_dir, file_name
        )
        train_image.save(train_filename)
        validation_image.save(validation_filename)

    def _create_plot_figures(self):
        self.accuracy_fig = MatPlot(
            "Metric",
            ("epoch", self.metric),
            ["train", "validation"],
            self._learning_dir
        )
        self.loss_fig = MatPlot(
            "Loss",
            ("epoch", "loss"),
            ["train", "validation"],
            self._learning_dir
        )
        if self.task == Task.SEMANTIC_SEGMENTATION:
            self.iou_fig = MatPlot(
                "IoU",
                ("epoch", "iou"),
                self.class_names,
                self._learning_dir
            )

    def on_epoch_end(self, epoch, logs={}):
        # post to milk
        milk_id = self.config['project_settings'].get('id')
        if milk_id:
            history = dict(
                train_config_id=int(milk_id),
                epoch_num=epoch,
                metric=float(logs.get(self.metric)),
                val_metric=float(logs.get('val_{}'.format(self.metric))),
                loss=float(logs.get('loss')),
                val_loss=float(logs.get('val_loss'))
            )
            farmer_res = self._milk_client.post(
                params=history,
                route='update_history'
            )
            if farmer_res.get('train_stopped'):
                self.model.stop_training = True
        # update learning figure
        self.accuracy_fig.add([logs.get(self.metric), logs.get(
            'val_{}'.format(self.metric))], is_update=True)
        self.loss_fig.add(
            [logs.get('loss'), logs.get('val_loss')], is_update=True)
        if self.task == Task.SEMANTIC_SEGMENTATION:
            self.iou_fig.add(self.iou_validation(
                self.validation_files, self.model), is_update=True)

        # display sample predict
        if epoch % 3 == 0:
            # for segmentation evaluation
            if self.task == Task.SEMANTIC_SEGMENTATION:
                train_set = self.generate_sample_result(
                    self.model,
                    self.train_files,
                    self.nb_classes,
                    self.height,
                    self.width
                )
                validation_set = self._generate_sample_result(
                    self.model,
                    self.validation_files,
                    self.nb_classes,
                    self.height,
                    self.width
                )
                self.save_image_from_ndarray(
                    train_set, validation_set, palette.palettes, epoch)

            if len(self.secret_config.sections()) > 0:
                secret_data = self.secret_config['default']
                if self.task == Task.SEMANTIC_SEGMENTATION:
                    file_name = os.path.join(self._learning_dir, 'IoU.png')
                else:
                    file_name = os.path.join(self._learning_dir, 'Metric.png')
                slack_logging(
                    file_name=file_name,
                    token=secret_data.get('slack_token'),
                    channel=secret_data.get('slack_channel'),
                    title=self.model_name
                )

    def on_train_end(self, logs=None):
        self._milk_client.close_session()
        self.model.save(os.path.join(self.model_dir, 'last_model.h5'))
        # evaluate on test data
        if self.task == Task.SEMANTIC_SEGMENTATION:
            last_model = build_model(
                task=self.task,
                model_name=self.model_name,
                nb_classes=self.nb_classes,
                height=self.height,
                width=self.width,
                backbone=self.backbone
            )
            last_model.load_weights(
                os.path.join(self.model_dir, 'best_model.h5')
            )
            test_ious = self.iou_validation(self.test_files, last_model)
            self.config['TEST'] = dict()
            for test_iou, class_name in zip(test_ious, self.class_names):
                self.config['TEST'][class_name] = str(test_iou)
            self.save_params(self._parameter)

    def iou_validation(self, data_set, model):
        conf = np.zeros((self.nb_classes, self.nb_classes), dtype=np.int32)
        print('IoU validation...')
        for image_file, seg_file in tqdm(data_set):
            # Get a training sample and make a prediction using current model
            sample = self.image_util.read_image(image_file, anti_alias=True)
            target = self.image_util.read_image(seg_file, normalization=False)
            predicted = np.asarray(model.predict_on_batch(
                np.expand_dims(sample, axis=0)))[0]

            # Convert predictions and target from categorical to integer format
            predicted = np.argmax(predicted, axis=-1).ravel()
            target = target.ravel()
            x = predicted + self.nb_classes * target
            bincount_2d = np.bincount(
                x.astype(np.int32), minlength=self.nb_classes**2)
            assert bincount_2d.size == self.nb_classes**2
            conf += bincount_2d.reshape((self.nb_classes, self.nb_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / \
                (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0

        return iou
