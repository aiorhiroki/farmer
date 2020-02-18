import segmentation_models

import torch
import torch.nn as nn
import torch.optim as optim

# The following line is needed to install segmentation-models-pytorch as a Python library
import segmentation_models_pytorch as smp

from segmentation_models import Unet, PSPNet
from segmentation_models import metrics
from segmentation_models.losses import (
    dice_loss, jaccard_loss, categorical_focal_loss, categorical_crossentropy
)

from farmer.ncc.models import xception, mobilenet, Deeplabv3, Model2D
from ..model.task_model import Task

from tensorflow import keras

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
                    # It couldn't find PyTorch branch
                    model = Model2D(nb_classes, height, width)

            elif self.config.framework == 'pytorch':
                print(
                    '[_do_make_model_task, Task.CLASSIFICATION][pytorch] under construction.'
                )

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
                if model_name == "unet":
                    model = smp.Unet(
                        backbone_name=backbone,
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                    )
                elif model_name == "deeplab_v3":
                    model = smp.Deeplabv3(
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                        backbone=backbone,
                    )
                elif model_name == "pspnet":
                    model = smp.PSPNet(
                        backbone_name=backbone,
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                    )
        else:
            raise NotImplementedError

        return model

    def _do_load_model_task(self, model, trained_model_path):
        if self.config.framework == "tensorflow":
            if trained_model_path:
                model.load_weights(trained_model_path)

        # if trained_model_path:
        #     if self.config.framework == "tensorflow":
        #         model.load_weights(trained_model_path)

        #     elif self.config.framework == 'pytorch':
        #         model.load_state_dict(
        #             torch.load(trained_model_path)
        #         )

        return model

    def _do_multi_gpu_task(self, base_model, multi_gpu, nb_gpu):
        if multi_gpu:
            if self.config.framework == "tensorflow":
                model = keras.utils.multi_gpu_model(base_model, gpus=nb_gpu)

            elif self.config.framework == 'pytorch':
                model = nn.DataParallel(base_model, device_ids=nb_gpu)

        else:
            model = base_model
        return model

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

        if self.config.framework == "tensorflow":
            if optimizer == "adam":
                optimizer = keras.optimizers.Adam(
                    lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
                )
            else:
                optimizer = keras.optimizers.SGD(
                    lr=learning_rate, momentum=0.9, decay=0.001
                )

            if task_id == Task.CLASSIFICATION:
                model.compile(
                    optimizer=optimizer,
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["acc"],
                )
            elif task_id == Task.SEMANTIC_SEGMENTATION:
                print('------------------')
                print('Loss:', loss_func)
                print('------------------')
                model.compile(
                    optimizer=optimizer,
                    loss=globals()[loss_func],
                    metrics=[metrics.iou_score,
                             categorical_crossentropy],
                )
            else:
                raise NotImplementedError

        elif self.config.framework == "pytorch":
            if optimizer == "adam":
                optimizer = optim.Adam(
                    lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
                )
            else:
                optimizer = optim.SGD(
                    lr=learning_rate, momentum=0.9, decay=0.001
                )

            if task_id == Task.CLASSIFICATION:
                model.compile(
                    optimizer=optimizer,
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["acc"],
                )

                # optimizer,
                # loss = nn.CrossEntropyLoss()
                # metrics...?

            elif task_id == Task.SEMANTIC_SEGMENTATION:
                print('------------------')
                print('Loss:', loss_func)
                print('------------------')
                model.compile(
                    optimizer=optimizer,
                    loss=globals()[loss_func],
                    metrics=[metrics.iou_score,
                             categorical_crossentropy],
                )

                # optimizer,
                # loss = nn.CrossEntropyLoss()
                # metrics...?

            else:
                raise NotImplementedError

        return model
