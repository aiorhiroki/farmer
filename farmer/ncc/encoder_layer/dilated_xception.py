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


TF_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

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

def Xception(nb_classes=None, input_tensor=None, input_shape=(512, 512, 3), weights='imagenet', OS=16, return_skip=False, include_top=True):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        return_skip: flag to return additional tensor after 2 SepConvs for decoder

    # Returns
        A Keras model instance.
    """
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

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax')(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='xception')

    # Load weights.
    # print("weights at dilated_xception: ", weights)
    if weights == 'imagenet':
        if include_top:
            weights_path = data_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
        else:
            weights_path = data_utils.get_file(
                'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                TF_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='b0042744bf5b25fce3cb969f33bebb97')
        model.load_weights(weights_path)
    elif not (weights in {'pascal_voc', 'cityscapes', None}):
        model.load_weights(weights)

    if return_skip:
        return model, skip
    else:
        return model

    

