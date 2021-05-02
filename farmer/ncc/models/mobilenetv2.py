import os
import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import data_utils

from .functional import relu6


BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def MobileNetV2(classes=1000, input_tensor=None, input_shape=(512, 512, 3), weights_info=None, OS=16, alpha=1., include_top=True):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        classes: Integer, optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        weights_info: this dict is consisted of `classes` and `weghts`.
            `classes` is number of `weights` output units.
            `weights` is one of 'imagenet' (pre-training on ImageNet), 'pascal_voc', 'cityscapes',
            original weights path (pre-training on original data) or None (random initialization)
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
        include_top: Boolean, whether to include the fully-connected
            layer at the top of the network. Defaults to `True`.

    # Returns
        A Keras model instance.

    """

    if weights_info is not None:
        if weights_info.get("weights") is None:
            weights = None

        elif weights_info["weights"] in {'imagenet', 'pascal_voc', 'cityscapes', None}:
            weights = weights_info["weights"]
        
        elif os.path.exists(weights_info["weights"]) and weights_info.get("classes") is not None:
            classes = int(weights_info["classes"])
            weights = weights_info["weights"]
        
        else:
            raise ValueError('The `weights` should be either '
                            '`None` (random initialization), `imagenet`, `pascal_voc`, `cityscapes`, '
                            'original weights path (pre-training on original data), '
                            'or the path to the weights file to be loaded and'
                            '`classes` should be number of original weights output units')

    else:
        weights = 'imagenet'
        if classes is None:
            raise ValueError('`classes` should be any number')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor
    
    # If input_shape is None, infer shape from input_tensor
    if backend.image_data_format() == 'channels_first':
        rows = input_shape[1]
        cols = input_shape[2]
    else:
        rows = input_shape[0]
        cols = input_shape[1]

    if rows == cols and rows in [96, 128, 160, 192, 224]:
        default_size = rows
    else:
        default_size = 224
    
    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                            'alpha can be one of `0.35`, `0.50`, `0.75`, '
                            '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224

    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
                kernel_size=3,
                strides=(2, 2), padding='same',
                use_bias=False, name='Conv')(img_input)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)
    
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    # Create model.
    model = Model(inputs, x, name='mobilenetv2')

    # Load weights.
    if weights == 'imagenet':
        print("movilenetv2 load model imagenet")
        model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                        str(alpha) + '_' + str(rows) + '.h5')
        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = data_utils.get_file(
            model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif not (weights in {'pascal_voc', 'cityscapes', None}):
        model.load_weights(weights)

    if include_top:
        return model
    else:
        # get last _inverted_res_block layer
        no_top_model = Model(
            inputs=model.input, 
            outputs=model.get_layer(index=-6).output
        )
        return no_top_model


def mobilenet_v2(nb_classes, height=244, width=244, weights_info=None):
    base_model = MobileNetV2(
        input_shape=(height, width, 3),
        weights_info=weights_info,
        include_top=False
    )
    x = base_model.output
    x = Conv2D(
        1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    return model
