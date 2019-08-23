import matplotlib.pyplot as plt
from ncc.readers import search_image_profile
from ncc.utils import palette
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
import random as rn
import multiprocessing as mp
from .model import build_model
from .image_util import ImageUtil
from .milk_client import MilkClient
from PIL import Image
import numpy as np
import requests
import datetime
import os
from tqdm import tqdm
from glob import glob
from configparser import ConfigParser
from farmer.ImageAnalyzer.task import Task
import csv


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
    IMAGE_PREFIX = "epoch_"
    IMAGE_EXTENSION = ".png"

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
        self._plot_manager = MatPlotManager(self._learning_dir)
        self.accuracy_fig = self.create_figure(
            "Metric", ("epoch", self.metric), ["train", "validation"]
        )
        self.loss_fig = self.create_figure(
            "Loss", ("epoch", "loss"), ["train", "validation"]
        )
        if self.task == Task.SEMANTIC_SEGMENTATION:
            self.iou_fig = self.create_figure(
                "IoU", ("epoch", "iou"), self.class_names
            )

        self.image_util = ImageUtil(self.nb_classes, (self.height, self.width))

        if self.milk_id and training:
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

    def _set_config_variables(self, config):
        # configに入っている値をインスタンス変数にする。
        self.config = config
        config_params = self.config['project_settings']
        self.epoch = int(config_params.get('epoch'))
        self.batch_size = int(config_params.get('batch_size'))
        self.learning_rate = float(config_params.get('learning_rate'))
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

        self.task = int(config_params.get('task_id'))
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

    def _save_image(self, train, validation, epoch):
        file_name = self.IMAGE_PREFIX + str(epoch) + self.IMAGE_EXTENSION
        train_filename = os.path.join(self._image_train_dir, file_name)
        validation_filename = os.path.join(
            self._image_validation_dir, file_name)
        train.save(train_filename)
        validation.save(validation_filename)

    def save_image_from_ndarray(self, train_set, validation_set,
                                palette, epoch, index_void=None):
        assert len(train_set) == len(validation_set) == 3
        train_image = Reporter.get_imageset(
            train_set[0], train_set[1], train_set[2], palette, index_void)
        validation_image = Reporter.get_imageset(
            validation_set[0], validation_set[1], validation_set[2],
            palette, index_void
        )
        self._save_image(train_image, validation_image, epoch)

    def create_figure(self, title, xy_labels, labels, filename=None):
        return self._plot_manager.add_figure(title, xy_labels,
                                             labels, filename)

    @staticmethod
    def concat_images(im1, im2, palette, mode):
        if mode == "P":
            assert palette is not None
            dst = Image.new("P", (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            dst.putpalette(palette)
        elif mode == "RGB":
            dst = Image.new("RGB", (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
        else:
            raise NotImplementedError

        return dst

    # index_void: 境界線のindexで学習・可視化の際は背景色と同じにする。
    @staticmethod
    def cast_to_pil(ndarray, palette, index_void=None):
        assert len(ndarray.shape) == 3
        res = np.argmax(ndarray, axis=2)
        if index_void is not None:
            res = np.where(res == index_void, 0, res)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette)
        return image

    @staticmethod
    def get_imageset(
        image_in_np,
        image_out_np,
        image_gt_np,
        palette,
        index_void=None
    ):
        image_out = Reporter.cast_to_pil(
            image_out_np, palette, index_void
        )
        image_tc = Reporter.cast_to_pil(
            image_gt_np, palette, index_void
        )
        image_merged = Reporter.concat_images(
            image_out, image_tc, palette, "P"
        ).convert("RGB")
        image_in_pil = Image.fromarray(
            np.uint8(image_in_np * 255), mode="RGB"
        )
        image_result = Reporter.concat_images(
            image_in_pil, image_merged, None, "RGB"
        )
        return image_result

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
                train_set = self._generate_sample_result()
                validation_set = self._generate_sample_result(training=False)
                self.save_image_from_ndarray(
                    train_set, validation_set, palette.palettes, epoch)

            if len(self.secret_config.sections()) > 0:
                self._slack_logging()

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

    def _slack_logging(self):
        if os.path.exists(os.path.join(self._learning_dir, 'IoU.png')):
            file_name = os.path.join(self._learning_dir, 'IoU.png')
        else:
            file_name = os.path.join(self._learning_dir, 'Metric.png')
        files = {'file': open(file_name, 'rb')}
        param = dict(
            token=self.secret_config.get('default', 'slack_token'),
            channels=self.secret_config.get('default', 'slack_channel'),
            filename="Metric Figure",
            title=self.model_name
        )
        requests.post(url='https://slack.com/api/files.upload',
                      params=param, files=files)

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

    def _generate_sample_result(self, training=True):
        if training:
            file_length = len(self.train_files)
        else:
            file_length = len(self.validation_files)

        random_index = np.random.randint(file_length)

        if training:
            sample_image_path = self.train_files[random_index]
        else:
            sample_image_path = self.validation_files[random_index]

        sample_image = self.image_util.read_image(
            sample_image_path[0],
            anti_alias=True
        )
        segmented = self.image_util.read_image(
            sample_image_path[1],
            normalization=False
        )
        sample_image, segmented = self._process_input(sample_image, segmented)
        output = self.model.predict(np.expand_dims(sample_image, axis=0))

        return [sample_image, output[0], segmented]

    def _process_input(self, images_original, labels):
        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint8)
        images_segmented = self.image_util.cast_to_onehot(labels)

        return images_original, images_segmented


# 図の保持
class MatPlotManager:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._figures = {}

    def add_figure(self, title, xy_labels, labels, filename=None):
        assert not(title in self._figures.keys()), "This title already exists."
        self._figures[title] = MatPlot(
            title, xy_labels, labels, self._root_dir, filename=filename)
        return self._figures[title]

    def get_figure(self, title):
        return self._figures[title]


# 学習履歴のプロット
class MatPlot:
    EXTENSION = ".png"

    def __init__(self, title, xy_labels, labels, root_dir, filename=None):
        assert len(labels) > 0 and len(xy_labels) == 2
        if filename is None:
            self._filename = title
        else:
            self._filename = filename
        self._title = title
        self._x_label, self._y_label = xy_labels[0], xy_labels[1]
        self._labels = labels
        self._root_dir = root_dir
        self._series = np.zeros((len(labels), 0))

    def add(self, series, is_update=False):
        series = np.asarray(series).reshape((len(series), 1))
        assert series.shape[0] == self._series.shape[0], 'not same length'
        self._series = np.concatenate([self._series, series], axis=1)
        if is_update:
            self.save()

    def save(self):
        plt.cla()
        for s, l in zip(self._series, self._labels):
            plt.plot(s, label=l)
        plt.legend()
        plt.grid()
        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.title(self._title)
        plt.savefig(os.path.join(self._root_dir,
                                 self._filename+self.EXTENSION))
