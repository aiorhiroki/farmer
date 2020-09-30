import segmentation_models

from farmer.ncc.optimizers import AdaBound
from farmer.ncc import losses, models
from ..model.task_model import Task

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa


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
            activation=self.config.train_params.activation,
            weights_info=self.config.train_params.weights_info,
            curriculum=self.config.curriculum,
            curriculum_model_path=self.config.curriculum_model_path,
        )
        model = self._do_compile_model_task(
            model,
            self.config.train_params.optimizer,
            self.config.train_params.learning_rate,
            self.config.task,
            self.config.train_params.loss,
            self.config.curriculum,
            self.config.curriculum_model_path,
        )
        model = self._do_load_model_task(
            model,
            self.config.trained_model_path,
            self.config.curriculum,
            self.config.curriculum_model_path,
            self.config.train_params.loss,
            self.config.task,
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
        activation="softmax",
        weights_info=None,
        curriculum=False,
        curriculum_model_path=None,
    ):
        if curriculum and curriculum_model_path:
            return
        if task == Task.CLASSIFICATION:
            xception_shape_condition = height >= 71 and width >= 71
            mobilenet_shape_condition = height >= 32 and width >= 32

            if model_name == "xception" and xception_shape_condition:
                model = models.xception(
                    nb_classes=nb_classes,
                    height=height,
                    width=width
                )
            elif model_name == "dilated_xception" and xception_shape_condition:
                model = models.dilated_xception(
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                    weights_info=weights_info
                )
            elif model_name == "mobilenet" and mobilenet_shape_condition:
                model = models.mobilenet(
                    nb_classes=nb_classes,
                    height=height,
                    width=width
                )
            elif model_name == "mobilenetv2" and mobilenet_shape_condition:
                model = models.mobilenet_v2(
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                    weights_info=weights_info
                )
            elif model_name.startswith("efficientnetb"):
                model = models.EfficientNet(
                    model_name=model_name,
                    nb_classes=nb_classes,
                    height=height,
                    width=width,
                )
            elif model_name.startswith('resnest'):
                model = models.resnest(
                    nb_classes=nb_classes,
                    model_name=model_name,
                    height=height,
                    width=width,
                )
            else:
                model = models.Model2D(nb_classes, height, width)

        elif task == Task.SEMANTIC_SEGMENTATION:
            print('------------------')
            print('Model:', model_name)
            print('Backbone:', backbone)
            print('------------------')

            if model_name == "unet":
                model = segmentation_models.Unet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
            elif model_name == "deeplab_v3":
                model = models.Deeplabv3(
                    weights_info=weights_info,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                    backbone=backbone,
                    activation=activation
                )
            elif model_name == "pspnet":
                model = segmentation_models.PSPNet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
            elif model_name == "fpn":
                model = segmentation_models.FPN(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
        else:
            raise NotImplementedError

        return model

    def _do_load_model_task(
        self,
        model,
        trained_model_path,
        curriculum,
        curriculum_model_path,
        loss_funcs,
        task_id,
    ):
        if trained_model_path:
            model.load_weights(trained_model_path)
        elif curriculum and curriculum_model_path:
            loss, loss_name = self._do_select_loss_task(
                loss_funcs=loss_funcs["functions"],
                task_id=task_id,
            )
            custom_objects = {
                loss_name:loss,
                'iou_score': segmentation_models.metrics.IOUScore,
                'f1-score': segmentation_models.metrics.FScore,
            }
            model = keras.models.load_model(
                curriculum_model_path,
                custom_objects=custom_objects
            )
        return model

    def _do_compile_model_task(
        self,
        model,
        optimizer,
        learning_rate,
        task_id,
        loss,
        curriculum=False,
        curriculum_model_path=None,
    ):
        if curriculum and curriculum_model_path:
            return
        if self.config.framework == "tensorflow":
            print('------------------')
            print('Optimizer:', optimizer)
            print('------------------')
            if optimizer == "adam":
                optimizer = keras.optimizers.Adam(
                    lr=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    decay=self.config.train_params.opt_decay,
                )
            elif optimizer == "adabound":
                optimizer = AdaBound(
                    learning_rate=learning_rate,
                    final_lr=0.1,
                )
            elif optimizer == "adamw":
                optimizer = tfa.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=self.config.train_params.opt_decay
                )
            elif optimizer == "radam":
                steps_per_epoch = self.config.nb_train_data // self.config.train_params.batch_size

                optimizer = tfa.optimizers.RectifiedAdam(
                    lr=learning_rate,
                    weight_decay=1e-5,
                    total_steps=int(steps_per_epoch * self.config.epochs * 0.95),
                    warmup_proportion=0.1,
                    min_lr=learning_rate * 0.01,
                )

                # Lookahead
                # https://arxiv.org/abs/1907.08610v1
                optimizer = tfa.optimizers.Lookahead(
                    optimizer,
                    sync_period=6,
                    slow_step_size=0.5
                )
            else:
                optimizer = keras.optimizers.SGD(
                    lr=learning_rate,
                    momentum=0.9,
                    decay=self.config.train_params.opt_decay
                )

            loss_funcs = loss["functions"]
            print('------------------')
            print('Loss:', loss_funcs.keys())
            print('------------------')
            loss, _ = self._do_select_loss_task(loss_funcs, task_id)
            if task_id == Task.CLASSIFICATION:
                metrics = ["acc"]
            elif task_id == Task.SEMANTIC_SEGMENTATION:
                metrics = [
                    segmentation_models.metrics.IOUScore(
                        class_indexes=list(range(1, self.config.nb_classes))),
                    segmentation_models.metrics.FScore(
                        class_indexes=list(range(1, self.config.nb_classes)))
                    ],

            model.compile(optimizer, loss, metrics)
        return model

    def _do_select_loss_task(self, loss_funcs, task_id):
        loss = None
        loss_name = None
        if task_id == Task.CLASSIFICATION:
            for i, loss_func in enumerate(loss_funcs.items()):
                loss_name, params = loss_func
                if i == 0:
                    if params is None:
                        loss = getattr(keras.losses, loss_name)()
                    else:
                        loss = getattr(keras.losses, loss_name)(**params)
                else:
                    if params is None:
                        loss += getattr(keras.losses, loss_name)()
                    else:
                        loss += getattr(keras.losses, loss_name)(**params)

        elif task_id == Task.SEMANTIC_SEGMENTATION:
            for i, loss_func in enumerate(loss_funcs.items()):
                loss_name, params = loss_func
                if params is not None and params.get("class_weights"):
                    params["class_weights"] = list(
                        params["class_weights"].values())
                if i == 0:
                    if params is None:
                        loss = getattr(losses, loss_name)()
                    else:
                        loss = getattr(losses, loss_name)(**params)
                else:
                    if params is None:
                        loss += getattr(losses, loss_name)()
                    else:
                        loss += getattr(losses, loss_name)(**params)
        
        return loss, loss.name
