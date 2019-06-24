from ncc.models import xception, Deeplabv3
from segmentation_models import Unet
from segmentation_models.losses import cce_dice_loss
from segmentation_models.metrics import iou_score


def build_model(
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone='resnet50'
):
    if task == 'classification':
        model = xception(nb_classes, width, height)
    elif task == 'segmentation':
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
