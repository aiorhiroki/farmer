from segmentation_models import Unet, PSPNet, FPN
from segmentation_models import metrics

from farmer.ncc.models import (
    xception, mobilenet, dilated_xception, mobilenet_v2, Deeplabv3, EfficientNet, resnest
)
from farmer.ncc.optimizers import AdaBound
from ..model.task_model import Task
from farmer.ncc import losses

from tensorflow import keras

class BuildModelTask:
    def __init__(self, config):
        self.config = config

    def command(self):
        # return: base_model is saved when training on multi gpu

        model = self._do_make_model_task(
            task=self.config.task,
            model_name=self.config.train_params.model_name,
            nb_classes=self.config.nb_classes,
            height=self.config.height,
            width=self.config.width,
            backbone=self.config.train_params.backbone,
            activation=self.config.train_params.activation
        )
        model = self._do_load_model_task(
            model, self.config.trained_model_path
        )
        model = self._do_compile_model_task(
            model,
            self.config.train_params.optimizer,
            self.config.train_params.learning_rate,
            self.config.task,
            self.config.train_params.loss
        )

        return model

    def _do_make_model_task(
        self,
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone="resnet50",
        activation="softmax"
    ):
        if task == Task.CLASSIFICATION:
            xception_shape_condition = height >= 71 and width >= 71
            mobilenet_shape_condition = height >= 32 and width >= 32

            if model_name == "xception" and xception_shape_condition:
                model = xception(
                    nb_classes=nb_classes,
                    height=height,
                    width=width
                )
            elif model_name == "dilated_xception" and xception_shape_condition:
                model = dilated_xception(
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                    weights_info=self.config.train_params.weights_info
                )
            elif model_name == "mobilenet" and mobilenet_shape_condition:
                model = mobilenet(
                    nb_classes=nb_classes,
                    height=height,
                    width=width
                )
            elif model_name == "mobilenetv2" and mobilenet_shape_condition:
                model = mobilenet_v2(
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                    weights_info=self.config.train_params.weights_info
                )
            elif model_name.startswith("efficientnetb"):
                model = EfficientNet(
                    model_name=model_name,
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                )
            elif model_name.startswith('resnest'):
                model = resnest(
                    nb_classes=nb_classes,
                    model_name=model_name,
                    height=height,
                    width=width,
                )
            else:
                model = Model2D(nb_classes, height, width)

        elif task == Task.SEMANTIC_SEGMENTATION:
            print('------------------')
            print('Model:', model_name)
            print('Backbone:', backbone)
            print('------------------')

            if model_name == "unet":
                model = Unet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
            elif model_name == "deeplab_v3":
                model = Deeplabv3(
                    weights_info=self.config.train_params.weights_info,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                    backbone=backbone,
                    activation=activation
                )
            elif model_name == "pspnet":
                model = PSPNet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
            elif model_name == "fpn":
                model = FPN(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
        else:
            raise NotImplementedError

        return model

    def _do_load_model_task(self, model, trained_model_path):
        if trained_model_path:
            model.load_weights(trained_model_path)
        return model

    def _do_compile_model_task(
        self,
        model,
        optimizer,
        learning_rate,
        task_id,
        loss_func
    ):
        if self.config.framework == "tensorflow":
            print('------------------')
            print('Optimizer:', optimizer)
            print('------------------')
            if optimizer == "adam":
                optimizer = keras.optimizers.Adam(
                    lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
                )
            elif optimizer == "adabound":
                optimizer = AdaBound(
                    learning_rate=learning_rate, final_lr=0.1
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
                loss_params = self.config.train_params.loss_params
                loss_params['class_weights'] = [
                    1.0 for i in range(self.config.nb_classes)]
                for cls_i, w in self.config.train_params.class_weights.items():
                    loss_params['class_weights'][cls_i] = w
                print('class weight:', loss_params['class_weights'])
                loss = getattr(losses, loss_func)(**loss_params)
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=[
                        metrics.IOUScore(
                            class_indexes=list(
                                range(1, self.config.nb_classes))),
                        metrics.FScore(
                            class_indexes=list(
                                range(1, self.config.nb_classes)))
                    ],
                )
            else:
                raise NotImplementedError

        return model
