from ncc.models import xception
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


def build_model(task, nb_classes, width=299, height=299, backbone='resnet50'):
    if task == 'classification':
        model = xception(nb_classes, width, height)
    elif task == 'segmentation':
        model = Unet(backbone, input_shape=(height, width, 3), classes=nb_classes)
    else:
        raise NotImplementedError

    return model


iou_score = iou_score
bce_jaccard_loss = bce_jaccard_loss
