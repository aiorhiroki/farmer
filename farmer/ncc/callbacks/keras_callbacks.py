import requests
import numpy as np
import os
import csv
import warnings
from ..utils import PostClient, MatPlotManager
from ..metrics import iou_dice_val, generate_segmentation_result

from tensorflow import keras


class BatchCheckpoint(keras.callbacks.Callback):
    # n batchごとに、モデルのsave & trainのaccとlossをcsvと画像で保存してslackに通知する
    def __init__(
        self,
        learning_path,
        filepath,
        token=None,
        channel=None,
        period=100
    ):
        self.learning_path = learning_path
        self.filepath = filepath
        self.period = period
        self.metric_filename = 'batch_metrics'
        self.metric_csv = f'{learning_path}/{self.metric_filename}.csv'
        self.title = f'train metrics every {self.period} batch'
        self.token = token
        self.channel = channel

        with open(self.metric_csv, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(['acc', 'loss'])
        self.plot_manager = MatPlotManager(self.learning_path)
        self.plot_manager.add_figure(
            title=self.title,
            xy_labels=(f"{self.period}*batch", 'metrics'),
            labels=[
                "acc",
                "loss"
            ],
            filename=self.metric_filename
        )

    def on_train_batch_end(self, batch, logs={}):
        if (batch + 1) % self.period != 0:
            return
        self.model.save(self.filepath)
        new_metric = [
            logs.get('acc'),
            logs.get('loss')
        ]
        with open(self.metric_csv, 'a') as fw:
            writer = csv.writer(fw)
            writer.writerow(new_metric)

        figure = self.plot_manager.get_figure(self.title)
        figure.add(new_metric, is_update=True)

        if self.token and self.channel:
            files = {
                'file': open(
                    f'{self.learning_path}/{self.metric_filename}.png', 'rb'
                )
            }
            param = dict(
                token=self.token,
                channels=self.channel,
                filename='metric figure',
                title=self.title
            )
            requests.post(
                url='https://slack.com/api/files.upload',
                params=param,
                files=files
            )


class GenerateSampleResult(keras.callbacks.Callback):
    def __init__(
        self,
        val_save_dir,
        valid_dataset,
        nb_classes,
        batch_size,
        segmentation_val_step=3
    ):
        self.val_save_dir = val_save_dir
        self.valid_dataset = valid_dataset
        self.nb_classes = nb_classes
        self.segmentation_val_step = segmentation_val_step
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        # display sample predict
        if (epoch + 1) % self.segmentation_val_step != 0:
            return

        save_dir = f"{self.val_save_dir}/epoch_{epoch + 1}"
        os.mkdir(save_dir)
        generate_segmentation_result(
            nb_classes=self.nb_classes,
            dataset=self.valid_dataset,
            model=self.model,
            save_dir=save_dir,
            batch_size=self.batch_size
        )


class SlackLogger(keras.callbacks.Callback):
    def __init__(
        self,
        logger_file,
        token,
        channel,
        title,
        filename='Metric Figure'
    ):
        """
        slackに画像ファイルを送信します。
        """
        self.logger_file = logger_file
        self.token = token
        self.channel = channel
        self.title = title
        self.filename = filename

    def on_epoch_end(self, epoch, logs={}):
        files = {'file': open(self.logger_file, 'rb')}
        param = dict(
            token=self.token,
            channels=self.channel,
            filename=self.filename,
            title=self.title
        )
        requests.post(
            url='https://slack.com/api/files.upload',
            params=param,
            files=files
        )


class PlotHistory(keras.callbacks.Callback):
    def __init__(self, save_dir, metrics):
        self.save_dir = save_dir
        self.metrics = metrics

    def on_train_begin(self, logs={}):
        self.plot_manager = MatPlotManager(self.save_dir)
        for metric in self.metrics:
            self.plot_manager.add_figure(
                title=metric,
                xy_labels=("epoch", metric),
                labels=[
                    "train",
                    "validation"
                ]
            )

    def on_epoch_end(self, epoch, logs={}):
        # update learning figure
        for metric in self.metrics:
            figure = self.plot_manager.get_figure(metric)
            figure.add(
                [
                    logs.get(metric),
                    logs.get('val_{}'.format(metric))
                ],
                is_update=True
            )


class IouHistory(keras.callbacks.Callback):
    def __init__(
        self,
        save_dir,
        valid_dataset,
        class_names,
        batch_size
    ):
        self.save_dir = save_dir
        self.valid_dataset = valid_dataset
        self.class_names = class_names
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.plot_manager = MatPlotManager(self.save_dir)
        self.plot_manager.add_figure(
            title="IoU",
            xy_labels=("epoch", "iou"),
            labels=self.class_names,
        )
        self.plot_manager.add_figure(
            title="Dice",
            xy_labels=("epoch", "dice"),
            labels=self.class_names,
        )

    def on_epoch_end(self, epoch, logs={}):
        iou_figure = self.plot_manager.get_figure("IoU")
        dice_figure = self.plot_manager.get_figure("Dice")

        nb_classes = len(self.class_names)
        iou_dice = iou_dice_val(
            nb_classes,
            self.valid_dataset,
            self.model,
            self.batch_size
        )
        iou_figure.add(
            iou_dice['iou'],
            is_update=True
        )
        dice_figure.add(
            iou_dice['dice'],
            is_update=True
        )


class PostHistory(keras.callbacks.Callback):
    def __init__(self, train_id, root_url, destination_url):
        self.train_id = train_id
        self.root_url = root_url
        self.destination_url = destination_url

    def on_train_begin(self, logs={}):
        self._client = PostClient(self.root_url)

    def on_epoch_end(self, epoch, logs={}):
        history = dict(
            train_id=int(self.train_id),
            epoch=epoch,
            acc=float(logs.get('acc')),
            val_acc=float(logs.get('val_acc')),
            loss=float(logs.get('loss')),
            val_loss=float(logs.get('val_loss'))
        )
        response = self._client.post(
            params=history,
            route=self.destination_url
        )
        if response.get('train_stopped'):
            self.model.stop_training = True

    def on_train_end(self, logs={}):
        self._client.close_session()


class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

# ==========================================================================
#
#  Multi-GPU Model Save Callback
#
# ==========================================================================


class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                'Epoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch + 1, self.monitor, self.best,
                                   current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(
                                filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)
