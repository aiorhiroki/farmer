from PIL import Image
import numpy as np
import requests
from configparser import ConfigParser
import datetime
import os
import csv
import matplotlib as mpl
mpl.use('Agg')  # to run this script by remote machine
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from ncc.readers import classification_set, segmentation_set


class Reporter(Callback):
    ROOT_DIR = "result"
    IMAGE_DIR = "image"
    LEARNING_DIR = "learning"
    INFO_DIR = "info"
    MODEL_DIR = "model"
    PARAMETER = "parameter.txt"
    TRAIN_FILE = "train_files.csv"
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
        self._image_test_dir = os.path.join(self._image_dir, "test")
        self._learning_dir = os.path.join(self._result_dir, self.LEARNING_DIR)
        self._info_dir = os.path.join(self._result_dir, self.INFO_DIR)
        self.model_dir = os.path.join(self._result_dir, self.MODEL_DIR)
        self._parameter = os.path.join(self._info_dir, self.PARAMETER)
        self.create_dirs()
        self.train_files = train_files
        self.test_files = test_files
        self._write_files(self.TRAIN_FILE, self.train_files)
        self._write_files(self.TEST_FILE, self.test_files)
        self.shuffle = shuffle
        self.task = task
        self.metric = 'acc' if self.task == 'classification' else 'iou'
        self.palette = self.get_palette()

        self.config = ConfigParser()
        self.config.read('config.ini')
        self.secret_config = ConfigParser()
        self.secret_config.read('secret.ini')

        self.epoch = self.config.getint('default', 'epoch')
        self.batch_size = self.config.getint('default', 'batch_size')
        self.optimizer = self.config.get('default', 'optimizer')
        self.augmentation = self.config.getboolean('default', 'augmentation')
        self.nb_classes = self.config.getint('default', 'nb_classes')
        self.model_name = self.config.get(task + '_default', 'model')
        self.height = self.config.getint(task + '_default', 'height')
        self.width = self.config.getint(task + '_default', 'width')
        self.backbone = self.config.get(task + '_default', 'backbone')

        self.save_params(self._parameter)

        self._plot_manager = MatPlotManager(self._learning_dir)
        self.accuracy_fig = self.create_figure("Metric", ("epoch", self.metric), ["train", "test"])
        self.loss_fig = self.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

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
        os.makedirs(self._image_test_dir)
        os.makedirs(self._learning_dir)
        os.makedirs(self._info_dir)
        os.makedirs(self.model_dir)

    def save_params(self, filename):
        self.config['Data'] = {'train files': len(self.train_files),
                               'test_files': len(self.test_files)}

        with open(filename, mode='w') as configfile:
            self.config.write(configfile)

    def read_annotation_set(self):
        target_dir = self.config.get('default', 'target_dir')
        train_dirs = self.config.get('default', 'train_dirs').split()
        test_dirs = self.config.get('default', 'test_dirs').split()

        if self.task == 'classification':
            train_set, test_set = classification_set(target_dir, train_dirs, test_dirs)
        elif self.task == 'segmentation':
            image_dir = self.config.get('segmentation_default', 'image_dir')
            label_dir = self.config.get('segmentation_default', 'label_dir')
            train_set, test_set = segmentation_set(target_dir, train_dirs, test_dirs, image_dir, label_dir)
        else:
            raise NotImplementedError

        return train_set, test_set

    def _save_image(self, train, test, epoch):
        file_name = self.IMAGE_PREFIX + str(epoch) + self.IMAGE_EXTENSION
        train_filename = os.path.join(self._image_train_dir, file_name)
        test_filename = os.path.join(self._image_test_dir, file_name)
        train.save(train_filename)
        test.save(test_filename)

    def save_image_from_ndarray(self, train_set, test_set, palette, epoch, index_void=None):
        assert len(train_set) == len(test_set) == 3
        train_image = Reporter.get_imageset(train_set[0], train_set[1], train_set[2], palette, index_void)
        test_image = Reporter.get_imageset(test_set[0], test_set[1], test_set[2], palette, index_void)
        self._save_image(train_image, test_image, epoch)

    def create_figure(self, title, xy_labels, labels, filename=None):
        return self._plot_manager.add_figure(title, xy_labels, labels, filename=filename)

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
    def get_imageset(image_in_np, image_out_np, image_gt_np, palette, index_void=None):
        assert image_in_np.shape[:2] == image_out_np.shape[:2] == image_gt_np.shape[:2]
        image_out, image_tc = Reporter.cast_to_pil(image_out_np, palette, index_void),\
                              Reporter.cast_to_pil(image_gt_np, palette, index_void)
        image_merged = Reporter.concat_images(image_out, image_tc, palette, "P").convert("RGB")
        image_in_pil = Image.fromarray(np.uint8(image_in_np * 255), mode="RGB")
        image_result = Reporter.concat_images(image_in_pil, image_merged, None, "RGB")
        return image_result

    def on_epoch_end(self, epoch, logs={}):
        # update learning figure
        self.accuracy_fig.add([logs.get(self.metric), logs.get('val_{}'.format(self.metric))], is_update=True)
        self.loss_fig.add([logs.get('loss'), logs.get('val_loss')], is_update=True)

        # display sample predict
        if epoch % 3 == 0:
            if self.task == 'segmentation':  # for segmentation evaluation
                train_set = self._generate_sample_result()
                test_set = self._generate_sample_result(training=False)
                self.save_image_from_ndarray(train_set, test_set, self.palette, epoch)

            self._slack_logging()

    def _slack_logging(self):
        files = {'file': open(os.path.join(self._learning_dir, 'Metric.png'), 'rb')}
        param = {
           'token': self.secret_config.get(self.CONFIG_SECTION, 'slack_token'),
           'channels': self.secret_config.get(self.CONFIG_SECTION, 'slack_channel'),
           'filename': "Metric Figure",
           'title': self.model_name
        }
        requests.post(url='https://slack.com/api/files.upload', params=param, files=files)

    def on_train_end(self, logs=None):
        self.model.save(self.model_dir + '/last_model.h5')

    def _generate_sample_result(self, training=True):
        file_length = len(self.train_files) if training else len(self.train_files)
        random_index = np.random.randint(file_length)
        sample_image_path = self.train_files[random_index]
        sample_image = self._read_image(sample_image_path[0], anti_alias=True)
        segmented = self._read_image(sample_image_path[1], normalization=False)
        sample_image, segmented = self._process_input(sample_image, segmented)
        output = self.model.predict(np.expand_dims(sample_image, axis=0))

        return [sample_image, output[0], segmented]

    def generate_batch_arrays(self, training=True):
        image_files = self.train_files if training else self.test_files
        if training and self.shuffle:
            np.random.shuffle(image_files)

        while True:
            x, y = [], []
            for image_file_set in image_files:
                input_file, label = image_file_set
                input_image = self._read_image(input_file, anti_alias=True)  # 入力画像は高品質にリサイズ

                if training and self.augmentation:
                    # not trained on out of category
                    # if np.sum(output_image) == 0:
                        # continue
                    # data augmentation
                    input_image, output_image = self.horizontal_flip(input_image)
                    input_image, output_image = self.vertical_flip(input_image)

                x.append(input_image)
                y.append(label)

                if len(x) == self.batch_size:
                    yield self._process_input(x, y)
                    x, y = [], []

    def _process_input(self, images_original, labels):
        # Cast to ndarray
        images_original = np.asarray(images_original, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint8)
        images_segmented = self.cast_to_onehot(labels)

        return images_original, images_segmented

    def _read_image(self, file_path, normalization=True, anti_alias=False):
        image = Image.open(file_path)
        # resize
        raw_size = (self.width, self.height)
        if raw_size != image.size:
            image = image.resize(raw_size, Image.ANTIALIAS) if anti_alias else image.resize(raw_size)
        # delete alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.asarray(image)
        if normalization:
            image = image / 255.0

        return image

    def cast_to_onehot(self, labels):
        if len(labels.shape) == 1:  # Classification
            one_hot = np.eye(self.nb_classes, dtype=np.uint8)
        else:  # Segmentation
            one_hot = np.identity(self.nb_classes, dtype=np.uint8)
        return one_hot[labels]

    @staticmethod
    def horizontal_flip(im1, im2=None, rate=0.5):
        if im2 is None:
            if np.random.rand() < rate:
                im1 = im1[:, ::-1, :]
            return im1
        else:
            if np.random.rand() < rate:
                im1 = im1[:, :, :]
                im2 = im2[:, ::-1]
            return im1, im2

    @staticmethod
    def vertical_flip(im1, im2=None, rate=0.5):
        if im2 is None:
            if np.random.rand() < rate:
                im1 = im1[::-1, :, :]
            return im1
        else:
            if np.random.rand() < rate:
                im1 = im1[::-1, :, :]
                im2 = im2[::-1, :]
            return im1, im2

    @staticmethod
    def random_crop(im1, im2, crop_size):
        h, w, _ = im1.shape

        if h - crop_size[0] == 0:
            top = 0
        else:
            top = np.random.randint(0, h - crop_size[0])

        if w - crop_size[1] == 0:
            left = 0
        else:
            left = np.random.randint(0, w - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        # 決めたtop, bottom, left, rightを使って画像を抜き出す
        im1 = im1[top:bottom, left:right, :]
        im2 = im2[top:bottom, left:right, :]

        return im1, im2


# 図の保持
class MatPlotManager:
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._figures = {}

    def add_figure(self, title, xy_labels, labels, filename=None):
        assert not(title in self._figures.keys()), "This title already exists."
        self._figures[title] = MatPlot(title, xy_labels, labels, self._root_dir, filename=filename)
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
        plt.savefig(os.path.join(self._root_dir, self._filename+self.EXTENSION))
