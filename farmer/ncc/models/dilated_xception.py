import os
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D

from .functional import SepConv_BN


TF_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual

    return outputs, skip


def DilatedXception(classes=10, input_tensor=None, input_shape=(512, 512, 3), weights_info=None, OS=16, return_skip=False, include_top=True):
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
        return_skip: flag to return additional tensor after 2 SepConvs for decoder
        include_top: Boolean, whether to include the fully-connected
            layer at the top of the network. Defaults to `True`.

    # Returns
        A Keras model instance. if return_skip is true, return additional tensor too
    """
    if weights_info is not None:
        if weights_info.get("weights") is None:
            weights = None

        elif weights_info["weights"] in {'pascal_voc', 'cityscapes', None}:
            weights = weights_info["weights"]
        
        elif os.path.exists(weights_info["weights"]):
            weights = weights_info["weights"]
            if weights_info.get("classes") is not None:
                classes = int(weights_info["classes"])
        
        else:
            raise ValueError('The `weights` should be either '
                            '`None` (random initialization), `pascal_voc`, `cityscapes`, '
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

    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)

    x = Conv2D(32, (3, 3), strides=(2, 2),
                name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x, _ = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False)

    x, _ = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x, _ = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x, _ = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x, _ = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='xception')

    # Load weights.
    if not (weights in {'imagenet', 'pascal_voc', 'cityscapes', None}):
        if weights_info.get("classes") is not None:
            model.load_weights(weights)

    # get model before FC layer
    if not include_top:
        model = Model(
            inputs=model.input, 
            outputs=model.get_layer(index=-3).output
        )
    
    if return_skip:
        return model, skip
    else:
        return model

def dilated_xception(nb_classes, height=512, width=512, weights_info=None):
    base_model = DilatedXception(
        classes=nb_classes,
        input_shape=(height, width, 3),
        weights_info=weights_info,
        include_top=False
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    return model

