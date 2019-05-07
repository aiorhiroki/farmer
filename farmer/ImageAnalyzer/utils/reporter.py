import matplotlib.pyplot as plt
from ncc.readers import classification_set, segmentation_set, data_set_from_annotation
from keras.callbacks import Callback
from keras.models import load_model
from .model import cce_dice_loss, iou_score
from .image_util import ImageUtil
from PIL import Image
import numpy as np
import requests
from configparser import ConfigParser
import datetime
import os
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
    CONFIG_SECTION = 'default'

    def __init__(self, task, shuffle=True, result_dir=None):
        super().__init__()
        if result_dir is None:
            result_dir = Reporter.generate_dir_name()
        self._root_dir = self.ROOT_DIR
        self._result_dir = os.path.join(self._root_dir, result_dir)
        self._image_dir = os.path.join(self._result_dir, self.IMAGE_DIR)
        self._image_train_dir = os.path.join(self._image_dir, "train")
        self._image_validation_dir = os.path.join(
            self._image_dir, "validation")
        self._learning_dir = os.path.join(self._result_dir, self.LEARNING_DIR)
        self._info_dir = os.path.join(self._result_dir, self.INFO_DIR)
        self.model_dir = os.path.join(self._result_dir, self.MODEL_DIR)
        self._parameter = os.path.join(self._info_dir, self.PARAMETER)
        self.create_dirs()
        self.shuffle = shuffle
        self.task = task
        self.metric = 'acc' if self.task == 'classification' else 'iou_score'
        self.palette = self.get_palette()

        self.config = ConfigParser()
        self.config.read('config.ini')
        self.secret_config = ConfigParser()
        self.secret_config.read('secret.ini')

        self.epoch = self.config.getint('default', 'epoch')
        self.batch_size = self.config.getint('default', 'batch_size')
        self.optimizer = self.config.get('default', 'optimizer')
        self.augmentation = self.config.getboolean('default', 'augmentation')
        self.gpu = self.config['default'].get('gpu') or None
        self.nb_split = self.config['default'].get('nb_split') or 1
        self.nb_classes = self.config.getint('project_settings', 'nb_classes')
        self.model_name = self.config.get(task + '_default', 'model')
        self.height = self.config.getint(task + '_default', 'height')
        self.width = self.config.getint(task + '_default', 'width')
        self.backbone = self.config.get(task + '_default', 'backbone')

        self.train_files, self.validation_files, self.test_files, self.class_names = self.read_annotation_set(
            task)
        self._write_files(self.TRAIN_FILE, self.train_files)
        self._write_files(self.VALIDATION_FILE, self.validation_files)
        self._write_files(self.TEST_FILE, self.test_files)

        self.config['Data'] = {'train files': len(self.train_files),
                               'validation_files': len(self.validation_files)}
        self.save_params(self._parameter)

        self._plot_manager = MatPlotManager(self._learning_dir)
        self.accuracy_fig = self.create_figure(
            "Metric", ("epoch", self.metric), ["train", "validation"])
        self.loss_fig = self.create_figure(
            "Loss", ("epoch", "loss"), ["train", "validation"])
        if task == 'segmentation':
            self.iou_fig = self.create_figure(
                "IoU", ("epoch", "iou"), self.class_names)
        self.image_util = ImageUtil(self.nb_classes, (self.height, self.width))

    def _write_files(self, csv_file, file_names):
        csv_path = os.path.join(self._info_dir, csv_file)
        with open(csv_path, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerows(file_names)

    @staticmethod
    def generate_dir_name():
        return datetime.datetime.today().strftime("%Y%m%d_%H%M")

    # カラーパレットを取得
    @staticmethod
    def get_palette():
        from ncc.utils import palette
        return palette.palettes

    def create_dirs(self):
        os.makedirs(self._root_dir, exist_ok=True)
        os.makedirs(self._result_dir)
        os.makedirs(self._image_dir)
        os.makedirs(self._image_train_dir)
        os.makedirs(self._image_validation_dir)
        os.makedirs(self._learning_dir)
        os.makedirs(self._info_dir)
        os.makedirs(self.model_dir)

    def save_params(self, filename):
        with open(filename, mode='w') as configfile:
            self.config.write(configfile)

    def read_annotation_set(self, task):
        target_dir = self.config.get(
            'project_settings', 'target_dir')
        train_data = self.config.get(
            'project_settings', 'train_data').split()
        validation_data = self.config.get(
            'project_settings', 'validation_data').split()
        test_data = self.config.get(
            'project_settings', 'test_data').split()

        class_names = None

        if len(train_data) == 1 and train_data[0].endswith('.csv'):
            train_set = data_set_from_annotation(train_data[0])
            validation_set = data_set_from_annotation(validation_data[0])
            if test_data:
                test_set = data_set_from_annotation(test_data[0])

        elif task == 'classification':
            train_set, class_names = classification_set(target_dir, train_data)
            validation_set, _ = classification_set(
                target_dir,
                validation_data,
                training=False,
                class_names=class_names
            )
            if test_data:
                test_set, _ = classification_set(
                    target_dir,
                    test_data,
                    training=False,
                    class_names=class_names
                )

        elif task == 'segmentation':
            image_dir = self.config.get('segmentation_default', 'image_dir')
            label_dir = self.config.get('segmentation_default', 'label_dir')
            train_set = segmentation_set(
                target_dir, train_data, image_dir, label_dir)
            validation_set = segmentation_set(
                target_dir, validation_data, image_dir, label_dir)
            if test_data:
                test_set = segmentation_set(
                    target_dir, test_data, image_dir, label_dir)
            class_names = self.config.get(
                'segmentation_default', 'class_names').split()
        else:
            raise NotImplementedError

        return train_set, validation_set, test_set, class_names

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
    def get_imageset(image_in_np, image_out_np, image_gt_np,
                     palette, index_void=None):
        assert image_in_np.shape[:2] == image_out_np.shape[:2] == image_gt_np.shape[:2]
        image_out, image_tc = Reporter.cast_to_pil(image_out_np, palette, index_void),\
            Reporter.cast_to_pil(image_gt_np, palette, index_void)
        image_merged = Reporter.concat_images(
            image_out, image_tc, palette, "P").convert("RGB")
        image_in_pil = Image.fromarray(np.uint8(image_in_np * 255), mode="RGB")
        image_result = Reporter.concat_images(
            image_in_pil, image_merged, None, "RGB")
        return image_result

    def on_epoch_end(self, epoch, logs={}):
        # update learning figure
        self.accuracy_fig.add([logs.get(self.metric), logs.get(
            'val_{}'.format(self.metric))], is_update=True)
        self.loss_fig.add(
            [logs.get('loss'), logs.get('val_loss')], is_update=True)
        if self.task == 'segmentation':
            self.iou_fig.add(self.iou_validation(
                self.validation_files, self.model), is_update=True)

        # display sample predict
        if epoch % 3 == 0:
            if self.task == 'segmentation':  # for segmentation evaluation
                train_set = self._generate_sample_result()
                validation_set = self._generate_sample_result(training=False)
                self.save_image_from_ndarray(
                    train_set, validation_set, self.palette, epoch)

            if len(self.secret_config.sections()) > 0:
                self._slack_logging()

    def iou_validation(self, data_set, model):
        conf = np.zeros((self.nb_classes, self.nb_classes), dtype=np.int32)

        for image_file, seg_file in data_set:
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

        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
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
        param = {
            'token': self.secret_config.get(self.CONFIG_SECTION, 'slack_token'),
            'channels': self.secret_config.get(self.CONFIG_SECTION, 'slack_channel'),
            'filename': "Metric Figure",
            'title': self.model_name
        }
        requests.post(url='https://slack.com/api/files.upload',
                      params=param, files=files)

    def on_train_end(self, logs=None):
        self.model.save(os.path.join(self.model_dir, 'last_model.h5'))
        last_model = load_model(
            os.path.join(self.model_dir, 'best_model.h5'),
            custom_objects={
                'cce_jaccard_loss': cce_dice_loss,
                'iou_score': iou_score}
        )
        # evaluate on test data
        if self.task == 'segmentation':
            test_ious = self.iou_validation(self.test_files, last_model)
            self.config['TEST'] = dict()
            for test_iou, class_name in zip(test_ious, self.class_names):
                self.config['TEST'][class_name] = test_iou
            self.save_params(self._parameter)

    def _generate_sample_result(self, training=True):
        file_length = len(self.train_files) if training else len(
            self.validation_files)
        random_index = np.random.randint(file_length)
        sample_image_path = self.train_files[random_index] if training else self.validation_files[random_index]
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
        assert series.shape[0] == self._series.shape[0], "series must have same length."
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
