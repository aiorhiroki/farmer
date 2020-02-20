import segmentation_models
import segmentation_models_pytorch as smp
from segmentation_models import Unet, PSPNet
from segmentation_models import metrics
from segmentation_models.losses import (
    dice_loss, jaccard_loss, categorical_focal_loss, categorical_crossentropy
)

from farmer.ncc.models import xception, mobilenet, Deeplabv3, Model2D
from ..model.task_model import Task

from tensorflow import keras

import torch
import torch.nn as nn
import torch.optim as optim

segmentation_models.set_framework('tf.keras')

# loss functions
cce_dice_loss = categorical_crossentropy + dice_loss
cce_jaccard_loss = categorical_crossentropy + jaccard_loss
categorical_focal_dice_loss = categorical_focal_loss + dice_loss
categorical_focal_jaccard_loss = categorical_focal_loss + jaccard_loss


class BuildModelTask:
    def __init__(self, config):
        self.config = config

    def command(self, trial=None):
        # return: base_model is saved when training on multi gpu

        base_model = self._do_make_model_task(
            task=self.config.task,
            model_name=self.config.model_name,
            nb_classes=self.config.nb_classes,
            height=self.config.height,
            width=self.config.width,
            backbone=self.config.backbone,
        )

        base_model = self._do_load_model_task(
            base_model, self.config.trained_model_path
        )

        model = self._do_multi_gpu_task(
            base_model, self.config.multi_gpu, self.config.nb_gpu
        )

        compiled_model = self._do_compile_model_task(
            model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.task,
            self.config.loss,
            trial
        )

        return compiled_model, base_model

    def _do_make_model_task(
        self,
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone="resnet50",
    ):
        if task == Task.CLASSIFICATION:
            if self.config.framework == 'tensorflow':
                xception_shape_condition = height >= 71 and width >= 71
                mobilenet_shape_condition = height >= 32 and width >= 32

                if model_name == "xception" and xception_shape_condition:
                    model = xception(nb_classes, height, width)
                elif model_name == "mobilenet" and mobilenet_shape_condition:
                    model = mobilenet(nb_classes, height, width)
                else:
                    model = Model2D(nb_classes, height, width)

            elif self.config.framework == 'pytorch':
                model = None

        elif task == Task.SEMANTIC_SEGMENTATION:
            print('------------------')
            print('Model:', model_name)
            print('Backbone:', backbone)
            print('------------------')

            if self.config.framework == "tensorflow":
                if model_name == "unet":
                    model = Unet(
                        backbone_name=backbone,
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                    )

                elif model_name == "deeplab_v3":
                    model = Deeplabv3(
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                        backbone=backbone,
                    )

                elif model_name == "pspnet":
                    model = PSPNet(
                        backbone_name=backbone,
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                    )

            elif self.config.framework == "pytorch":
                print('SEMANTIC_SEGMENTATION, pytorch condition')
                print(backbone, 3, nb_classes)

                if model_name == "unet":
                    print('fetch Unet model')
                    model = smp.Unet(
                        encoder_name=backbone,
                        in_channels=3,
                        classes=nb_classes,
                    )

                elif model_name == "pspnet":
                    model = smp.PSPNet(
                        backbone_name=backbone,
                        in_channels=3,
                        classes=nb_classes,
                    )

        else:
            raise NotImplementedError

        return model

    def _do_load_model_task(self, model, trained_model_path):
        if trained_model_path:
            if self.config.framework == "tensorflow":
                model.load_weights(trained_model_path)

            if self.config.framework == "pytorch":
                print('_do_load_model_task, pytorch, it is under construction.')

                state_dict = torch.load(trained_model_path)
                # state_dict = torch.load(trained_model_path, map_location=lambda storage, loc:storage)

                model.load_state_dict(state_dict)

        return model

    def _do_multi_gpu_task(self, base_model, multi_gpu, nb_gpu):
        if multi_gpu:
            if self.config.framework == "tensorflow":
                model = keras.utils.multi_gpu_model(base_model, gpus=nb_gpu)

            elif self.config.framework == "pytorch":
                model = nn.DataParallel(model, device_ids=nb_gpu)

        else:
            model = base_model

        return model

    def _do_fetch_optimizer_task(self, parameters, optimizer_name, learning_rate):
        if optimizer_name == "adam":
            if self.config.framework == "tensorflow":
                return keras.optimizers.Adam(
                    lr=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    decay=0.001,
                )

            elif self.config.framework == "pytorch":
                return optim.Adam(
                    params=parameters,
                    lr=learning_rate,
                    betas=(0.9, 0.999),
                    weight_decay=0.001,
                )

        else:
            if self.config.framework == "tensorflow":
                return keras.optimizers.SGD(
                    lr=learning_rate,
                    momentum=0.9,
                    decay=0.001,
                )

            elif self.config.framework == "pytorch":
                return optim.SGD(
                    params=parameters,
                    lr=learning_rate,
                    momentum=0.9,
                    weight_decay=0.001,
                )

    def _do_setup_model_task(self, task_id, model, optimizer, loss_func):
        if task_id == Task.CLASSIFICATION:
            if self.config.framework == "tensorflow":
                model.compile(
                    optimizer=optimizer,
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["acc"],
                )

                return model

            elif self.config.framework == "pytorch":
                return model

        elif task_id == Task.SEMANTIC_SEGMENTATION:
            if self.config.framework == "tensorflow":
                print('------------------')
                print('Loss:', loss_func)
                print('------------------')
                model.compile(
                    optimizer=optimizer,
                    loss=globals()[loss_func],
                    metrics=[metrics.iou_score,
                             categorical_crossentropy],
                )

                return model

            elif self.config.framework == "pytorch":
                return model

        else:
            raise NotImplementedError

    def _do_compile_model_task(
        self,
        model,
        optimizer,
        learning_rate,
        task_id,
        loss_func,
        trial
    ):
        if self.config.op_learning_rate:
            learning_rate = int(trial.suggest_discrete_uniform(
                'learning_rate', *self.config.learning_rate))
        else:
            learning_rate = self.config.learning_rate

        # if optimizer == "adam":

        #     optimizer = keras.optimizers.Adam(
        #         lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
        #     )
        # else:
        #     optimizer = keras.optimizers.SGD(
        #         lr=learning_rate, momentum=0.9, decay=0.001
        #     )

        optimizer = self._do_fetch_optimizer_task(
            model.parameters(),
            optimizer,
            learning_rate
        )

        # if task_id == Task.CLASSIFICATION:
        #     model.compile(
        #         optimizer=optimizer,
        #         loss=keras.losses.categorical_crossentropy,
        #         metrics=["acc"],
        #     )
        # elif task_id == Task.SEMANTIC_SEGMENTATION:
        #     print('------------------')
        #     print('Loss:', loss_func)
        #     print('------------------')
        #     model.compile(
        #         optimizer=optimizer,
        #         loss=globals()[loss_func],
        #         metrics=[metrics.iou_score,
        #                  categorical_crossentropy],
        #     )
        # else:
        #     raise NotImplementedError

        model = self._do_setup_model_task(
            task_id,
            model,
            optimizer,
            loss_func,
        )

        # if self.config.framework == "tensorflow":
        #     if optimizer == "adam":
        #         optimizer = keras.optimizers.Adam(
        #             lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
        #         )
        #     else:
        #         optimizer = keras.optimizers.SGD(
        #             lr=learning_rate, momentum=0.9, decay=0.001
        #         )

        #     if task_id == Task.CLASSIFICATION:
        #         model.compile(
        #             optimizer=optimizer,
        #             loss=keras.losses.categorical_crossentropy,
        #             metrics=["acc"],
        #         )
        #     elif task_id == Task.SEMANTIC_SEGMENTATION:
        #         print('------------------')
        #         print('Loss:', loss_func)
        #         print('------------------')
        #         model.compile(
        #             optimizer=optimizer,
        #             loss=globals()[loss_func],
        #             metrics=[metrics.iou_score,
        #                      categorical_crossentropy],
        #         )
        #     else:
        #         raise NotImplementedError

        # elif self.config.framework == "pytorch":
        #     print('_do_compile_model_task, pytorch, it is under construction.')
        #     # if optimizer == "adam":
        #     #     optimizer = keras.optimizers.Adam(
        #     #         lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
        #     #     )
        #     # else:
        #     #     optimizer = keras.optimizers.SGD(
        #     #         lr=learning_rate, momentum=0.9, decay=0.001
        #     #     )

        #     # if task_id == Task.CLASSIFICATION:
        #     #     model.compile(
        #     #         optimizer=optimizer,
        #     #         loss=keras.losses.categorical_crossentropy,
        #     #         metrics=["acc"],
        #     #     )
        #     # elif task_id == Task.SEMANTIC_SEGMENTATION:
        #     #     print('------------------')
        #     #     print('Loss:', loss_func)
        #     #     print('------------------')
        #     #     model.compile(
        #     #         optimizer=optimizer,
        #     #         loss=globals()[loss_func],
        #     #         metrics=[metrics.iou_score,
        #     #                  categorical_crossentropy],
        #     #     )
        #     # else:
        #     #     raise NotImplementedError

        return model
