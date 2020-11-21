from tensorflow.keras.layers import Reshape, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model

from .Deeplabv3 import Deeplabv3
from .functional import Subpixel, icnr_weights, do_crf


def Deeplabv3_subpixel(weights_info=None, input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
                       OS=16, alpha=1., activation='softmax', apply_crf=True, load_weights=False):
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

    x = Subpixel(classes, 1, scale, padding='same')(base_model.output)
    # x = Reshape((input_shape[0] * input_shape[1], -1))(x)
    x = Activation(activation, name='pred_mask')(x)

    if apply_crf:
        gt_pr = Concatenate()([base_model.input, x])
        x = Lambda(lambda x: do_crf(x[:, 0], x[:, 1]), output_shape=())(gt_pr)

    model = Model(base_model.input, x, name='deeplabv3p_subpixel')

    # # Do ICNR
    # for layer in model.layers:
    #     if type(layer) == Subpixel:
    #         c, b = layer.get_weights()
    #         w = icnr_weights(scale=scale, shape=c.shape)
    #         layer.set_weights([w, b])

    if load_weights and backbone == 'mobilenetv2':
        model.load_weights('weights/{}_{}.h5'.format(backbone, 'subpixcel'))

    return model
