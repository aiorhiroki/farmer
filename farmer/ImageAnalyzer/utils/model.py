from ncc.models import xception, mobilenet, Deeplabv3, Model2D
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
