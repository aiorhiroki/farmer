from ncc.models import xception, Deeplabv3, Model2D
from segmentation_models import Unet
from segmentation_models.losses import cce_dice_loss
from segmentation_models.metrics import iou_score
from farmer.ImageAnalyzer.task import Task


def build_model(
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone='resnet50'
):
    if task == Task.CLASSIFICATION:
        if height < 72 or width < 72:
            model = Model2D(
                input_shape=(height, width, 3),
                num_classes=nb_classes
            )
        else:
            model = xception(nb_classes, height, width)
    elif task == Task.SEMANTIC_SEGMENTATION:
        if model_name == "unet":
            model = Unet(
                backbone,
                input_shape=(height, width, 3),
                classes=nb_classes
            )
        elif model_name == "deeplabv3":
            model = Deeplabv3(
                input_shape=(height, width, 3),
                classes=nb_classes,
                backbone='xception'
            )
    else:
        raise NotImplementedError

    return model


iou_score = iou_score
cce_dice_loss = cce_dice_loss
