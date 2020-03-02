import segmentation_models as sm
import segmentation_models_pytorch as smp

from segmentation_models import metrics
from segmentation_models.losses import (
    dice_loss, jaccard_loss, categorical_focal_loss, categorical_crossentropy
)

from farmer.ncc.models import xception, mobilenet, Deeplabv3, Model2D
from ..model.task_model import Task

from tensorflow import keras
import torch

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

        compiled_model, optimizer = self._do_compile_model_task(
            model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.task,
            self.config.loss,
            trial
        )

        return compiled_model, base_model, optimizer

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
                    model = sm.Unet(
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
                    model = sm.PSPNet(
                        backbone_name=backbone,
                        input_shape=(height, width, 3),
                        classes=nb_classes,
                    )

            elif self.config.framework == "pytorch":
                if model_name == "unet":
                    model = smp.Unet(
                        encoder_name=backbone,
                        in_channels=3,
                        classes=nb_classes,
                    )

                elif model_name == "pspnet":
                    model = smp.PSPNet(
                        encoder_name=backbone,
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
                model.load_state_dict(
                    torch.load(trained_model_path)
                )

        return model

    def _do_multi_gpu_task(self, base_model, multi_gpu, nb_gpu):
        if multi_gpu:
            if self.config.framework == "tensorflow":
                model = keras.utils.multi_gpu_model(base_model, gpus=nb_gpu)

            elif self.config.framework == "pytorch":
                model = torch.nn.DataParallel(model, device_ids=nb_gpu)

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

        optimizer = self._do_fetch_optimizer_task(
            model,
            optimizer,
            learning_rate
        )

        model = self._do_setup_model_task(
            task_id,
            model,
            optimizer,
            loss_func,
        )

        return model, optimizer


    def _do_fetch_optimizer_task(self, model, optimizer_name, learning_rate):
        if optimizer_name == "adam":
            if self.config.framework == "tensorflow":
                return keras.optimizers.Adam(
                    lr=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    decay=0.001,
                )

            elif self.config.framework == "pytorch":
                return torch.optim.Adam(
                    params=model.parameters(),
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
                return torch.optim.SGD(
                    params=model.parameters(),
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