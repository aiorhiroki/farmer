from ncc.models import xception
from segmentation_models import Unet


def build_model(task, nb_classes, width=299, height=299, backbone='resnet50'):
    if task == 'classification':
        model = xception(nb_classes, width, height)
    elif task == 'segmentation':
        model = Unet(backbone, input_shape=(width, height, 3), classes=nb_classes)
    else:
        raise NotImplementedError

    return model

