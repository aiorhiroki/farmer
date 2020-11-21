from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model

from .Deeplabv3 import Deeplabv3
from .functional import Subpixel, ICNR


def Deeplabv3_subpixel(weights_info={}, input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
                       OS=16, alpha=1., activation='softmax'):
    '''
    original deeplab v3+ and subpixel upsampling layer
    '''

    model = Deeplabv3(weights_info=weights_info, input_tensor=input_tensor,
                      input_shape=input_shape, classes=classes,
                      backbone=backbone, OS=OS, alpha=alpha, activation=activation)

    base_model = Model(model.input, model.get_layer(index=-5).output)

    if backbone == 'xception':
        scale = 4
    else:
        scale = 8

    x = Subpixel(classes, 1, scale,
                 padding='same',
                 kernel_initializer=ICNR(scale=scale))(base_model.output)
    x = Activation(activation, name='pred_mask')(x)

    model = Model(base_model.input, x, name='deeplabv3p_subpixel')
    return model
