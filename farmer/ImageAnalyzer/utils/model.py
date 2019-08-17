from ncc.models import xception, mobilenet, Deeplabv3, Model2D
from segmentation_models import Unet
from segmentation_models.losses import cce_dice_loss
from segmentation_models.metrics import iou_score
from farmer.ImageAnalyzer.task import Task
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf


def build_model(reporter):
    with tf.device("/cpu:0"):
        base_model = make_model(
            task=reporter.task,
            model_name=reporter.model_name,
            nb_classes=reporter.nb_classes,
            height=reporter.height,
            width=reporter.width,
            backbone=reporter.backbone
        )

    if reporter.model_path is not None:
        base_model.load_weights(reporter.model_path)

    if reporter.multi_gpu:
        model = multi_gpu_model(base_model, gpus=reporter.nb_gpu)
    else:
        model = base_model

    compiled_model = compile_model(model)

    return compiled_model, base_model


def make_model(
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone='resnet50'
):
    if task == Task.CLASSIFICATION:
        if model_name == 'xception' and (height >= 71 and width >= 71):
            model = xception(nb_classes, height, width)
        elif model_name == 'mobilenet' and (height >= 32 and width >= 32):
            model = mobilenet(nb_classes, height, width)
        else:
            model = Model2D(
                input_shape=(height, width, 3),
                num_classes=nb_classes
            )

    elif task == Task.SEMANTIC_SEGMENTATION:
        if model_name == "unet":
            model = Unet(
                backbone,
                input_shape=(height, width, 3),
                classes=nb_classes
            )
        elif model_name == "deeplab_v3":
            model = Deeplabv3(
                input_shape=(height, width, 3),
                classes=nb_classes,
                backbone='xception'
            )
    else:
        raise NotImplementedError

    return model


def compile_model(
    model,
    optimizer,
    learning_rate,
    task_id
):

    if optimizer == 'adam':
        optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
        )
    else:
        optimizer = optimizers.SGD(
            lr=learning_rate, momentum=0.9, decay=0.001
        )

    if task_id == Task.CLASSIFICATION:
        model.compile(
            optimizer=optimizer,
            loss=categorical_crossentropy,
            metrics=['acc']
        )
    elif task_id == Task.SEMANTIC_SEGMENTATION:
        model.compile(
            optimizer=optimizer,
            loss=cce_dice_loss,
            metrics=[iou_score]
        )
    else:
        raise NotImplementedError

    return model
