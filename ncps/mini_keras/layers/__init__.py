"""Layer implementations for mini-keras.

This module provides the core layer implementations including RNN variants.
"""

from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.activations.activation import Activation
from ncps.mini_keras.layers.activations.elu import ELU
from ncps.mini_keras.layers.activations.leaky_relu import LeakyReLU
from ncps.mini_keras.layers.activations.prelu import PReLU
from ncps.mini_keras.layers.activations.relu import ReLU
from ncps.mini_keras.layers.activations.softmax import Softmax
from ncps.mini_keras.layers.attention.additive_attention import AdditiveAttention
from ncps.mini_keras.layers.attention.attention import Attention
from ncps.mini_keras.layers.attention.grouped_query_attention import (
    GroupedQueryAttention,
)
from ncps.mini_keras.layers.attention.multi_head_attention import MultiHeadAttention
from ncps.mini_keras.layers.convolutional.conv1d import Conv1D
from ncps.mini_keras.layers.convolutional.conv1d_transpose import Conv1DTranspose
from ncps.mini_keras.layers.convolutional.conv2d import Conv2D
from ncps.mini_keras.layers.convolutional.conv2d_transpose import Conv2DTranspose
from ncps.mini_keras.layers.convolutional.conv3d import Conv3D
from ncps.mini_keras.layers.convolutional.conv3d_transpose import Conv3DTranspose
from ncps.mini_keras.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from ncps.mini_keras.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from ncps.mini_keras.layers.convolutional.separable_conv1d import SeparableConv1D
from ncps.mini_keras.layers.convolutional.separable_conv2d import SeparableConv2D
from ncps.mini_keras.layers.core.dense import Dense
from ncps.mini_keras.layers.core.einsum_dense import EinsumDense
from ncps.mini_keras.layers.core.embedding import Embedding
from ncps.mini_keras.layers.core.identity import Identity
from ncps.mini_keras.layers.core.input_layer import Input
from ncps.mini_keras.layers.core.input_layer import InputLayer
from ncps.mini_keras.layers.core.lambda_layer import Lambda
from ncps.mini_keras.layers.core.masking import Masking
from ncps.mini_keras.layers.core.wrapper import Wrapper
from ncps.mini_keras.layers.input_spec import InputSpec
from ncps.mini_keras.layers.layer import Layer
from ncps.mini_keras.layers.merging.add import Add
from ncps.mini_keras.layers.merging.add import add
from ncps.mini_keras.layers.merging.average import Average
from ncps.mini_keras.layers.merging.average import average
from ncps.mini_keras.layers.merging.concatenate import Concatenate
from ncps.mini_keras.layers.merging.concatenate import concatenate
from ncps.mini_keras.layers.merging.dot import Dot
from ncps.mini_keras.layers.merging.dot import dot
from ncps.mini_keras.layers.merging.maximum import Maximum
from ncps.mini_keras.layers.merging.maximum import maximum
from ncps.mini_keras.layers.merging.minimum import Minimum
from ncps.mini_keras.layers.merging.minimum import minimum
from ncps.mini_keras.layers.merging.multiply import Multiply
from ncps.mini_keras.layers.merging.multiply import multiply
from ncps.mini_keras.layers.merging.subtract import Subtract
from ncps.mini_keras.layers.merging.subtract import subtract
from ncps.mini_keras.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from ncps.mini_keras.layers.normalization.group_normalization import (
    GroupNormalization,
)
from ncps.mini_keras.layers.normalization.layer_normalization import (
    LayerNormalization,
)
from ncps.mini_keras.layers.normalization.spectral_normalization import (
    SpectralNormalization,
)
from ncps.mini_keras.layers.normalization.unit_normalization import UnitNormalization
from ncps.mini_keras.layers.pooling.average_pooling1d import AveragePooling1D
from ncps.mini_keras.layers.pooling.average_pooling2d import AveragePooling2D
from ncps.mini_keras.layers.pooling.average_pooling3d import AveragePooling3D
from ncps.mini_keras.layers.pooling.global_average_pooling1d import (
    GlobalAveragePooling1D,
)
from ncps.mini_keras.layers.pooling.global_average_pooling2d import (
    GlobalAveragePooling2D,
)
from ncps.mini_keras.layers.pooling.global_average_pooling3d import (
    GlobalAveragePooling3D,
)
from ncps.mini_keras.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from ncps.mini_keras.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from ncps.mini_keras.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D
from ncps.mini_keras.layers.pooling.max_pooling1d import MaxPooling1D
from ncps.mini_keras.layers.pooling.max_pooling2d import MaxPooling2D
from ncps.mini_keras.layers.pooling.max_pooling3d import MaxPooling3D
from ncps.mini_keras.layers.preprocessing.category_encoding import CategoryEncoding
from ncps.mini_keras.layers.preprocessing.discretization import Discretization
from ncps.mini_keras.layers.preprocessing.hashed_crossing import HashedCrossing
from ncps.mini_keras.layers.preprocessing.hashing import Hashing
from ncps.mini_keras.layers.preprocessing.image_preprocessing.aug_mix import AugMix
from ncps.mini_keras.layers.preprocessing.image_preprocessing.auto_contrast import (
    AutoContrast,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.center_crop import (
    CenterCrop,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.equalization import (
    Equalization,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.max_num_bounding_box import (
    MaxNumBoundingBoxes,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.mix_up import MixUp
from ncps.mini_keras.layers.preprocessing.image_preprocessing.rand_augment import (
    RandAugment,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_brightness import (
    RandomBrightness,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_color_degeneration import (
    RandomColorDegeneration,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_color_jitter import (
    RandomColorJitter,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_contrast import (
    RandomContrast,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_crop import (
    RandomCrop,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_flip import (
    RandomFlip,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_grayscale import (
    RandomGrayscale,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_hue import (
    RandomHue,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_posterization import (
    RandomPosterization,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_rotation import (
    RandomRotation,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_saturation import (
    RandomSaturation,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_sharpness import (
    RandomSharpness,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_shear import (
    RandomShear,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_translation import (
    RandomTranslation,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.random_zoom import (
    RandomZoom,
)
from ncps.mini_keras.layers.preprocessing.image_preprocessing.resizing import Resizing
from ncps.mini_keras.layers.preprocessing.image_preprocessing.solarization import (
    Solarization,
)
from ncps.mini_keras.layers.preprocessing.index_lookup import IndexLookup
from ncps.mini_keras.layers.preprocessing.integer_lookup import IntegerLookup
from ncps.mini_keras.layers.preprocessing.mel_spectrogram import MelSpectrogram
from ncps.mini_keras.layers.preprocessing.normalization import Normalization
from ncps.mini_keras.layers.preprocessing.pipeline import Pipeline
from ncps.mini_keras.layers.preprocessing.rescaling import Rescaling
from ncps.mini_keras.layers.preprocessing.stft_spectrogram import STFTSpectrogram
from ncps.mini_keras.layers.preprocessing.string_lookup import StringLookup
from ncps.mini_keras.layers.preprocessing.text_vectorization import TextVectorization
from ncps.mini_keras.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from ncps.mini_keras.layers.regularization.alpha_dropout import AlphaDropout
from ncps.mini_keras.layers.regularization.dropout import Dropout
from ncps.mini_keras.layers.regularization.gaussian_dropout import GaussianDropout
from ncps.mini_keras.layers.regularization.gaussian_noise import GaussianNoise
from ncps.mini_keras.layers.regularization.spatial_dropout import SpatialDropout1D
from ncps.mini_keras.layers.regularization.spatial_dropout import SpatialDropout2D
from ncps.mini_keras.layers.regularization.spatial_dropout import SpatialDropout3D
from ncps.mini_keras.layers.reshaping.cropping1d import Cropping1D
from ncps.mini_keras.layers.reshaping.cropping2d import Cropping2D
from ncps.mini_keras.layers.reshaping.cropping3d import Cropping3D
from ncps.mini_keras.layers.reshaping.flatten import Flatten
from ncps.mini_keras.layers.reshaping.permute import Permute
from ncps.mini_keras.layers.reshaping.repeat_vector import RepeatVector
from ncps.mini_keras.layers.reshaping.reshape import Reshape
from ncps.mini_keras.layers.reshaping.up_sampling1d import UpSampling1D
from ncps.mini_keras.layers.reshaping.up_sampling2d import UpSampling2D
from ncps.mini_keras.layers.reshaping.up_sampling3d import UpSampling3D
from ncps.mini_keras.layers.reshaping.zero_padding1d import ZeroPadding1D
from ncps.mini_keras.layers.reshaping.zero_padding2d import ZeroPadding2D
from ncps.mini_keras.layers.reshaping.zero_padding3d import ZeroPadding3D
from ncps.mini_keras.layers.rnn.bidirectional import Bidirectional
from ncps.mini_keras.layers.rnn.conv_lstm1d import ConvLSTM1D
from ncps.mini_keras.layers.rnn.conv_lstm2d import ConvLSTM2D
from ncps.mini_keras.layers.rnn.conv_lstm3d import ConvLSTM3D
from ncps.mini_keras.layers.rnn.gru import GRU
from ncps.mini_keras.layers.rnn.gru import GRUCell
from ncps.mini_keras.layers.rnn.lstm import LSTM
from ncps.mini_keras.layers.rnn.lstm import LSTMCell
from ncps.mini_keras.layers.rnn.rnn import RNN
from ncps.mini_keras.layers.rnn.simple_rnn import SimpleRNN
from ncps.mini_keras.layers.rnn.simple_rnn import SimpleRNNCell
from ncps.mini_keras.layers.rnn.stacked_rnn_cells import StackedRNNCells
from ncps.mini_keras.layers.rnn.time_distributed import TimeDistributed
from ncps.mini_keras.saving import serialization_lib
from ncps.mini_keras.layers.rnn.abstract_rnn_cell import AbstractRNNCell


@keras_mini_export("ncps.mini_keras.layers.serialize")
def serialize(layer):
    """Returns the layer configuration as a Python dict.

    Args:
        layer: A `keras.layers.Layer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the layer.
    """
    return serialization_lib.serialize_keras_object(layer)


@keras_mini_export("ncps.mini_keras.layers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras layer object via its configuration.

    Args:
        config: A python dict containing a serialized layer configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras layer instance.
    """
    obj = serialization_lib.deserialize_keras_object(
        config,
        custom_objects=custom_objects,
    )
    if not isinstance(obj, Layer):
        raise ValueError(
            "`ncps.mini_keras.layers.deserialize` was passed a `config` object that is "
            f"not a `keras.layers.Layer`. Received: {config}"
        )
    return obj

__all__ = [ "Activation", "ELU", "LeakyReLU", "PReLU", "ReLU", "Softmax", "AdditiveAttention", "Attention", 
           "GroupedQueryAttention", "MultiHeadAttention", "Conv1D", "Conv1DTranspose", "Conv2D", "Conv2DTranspose", 
           "Conv3D", "Conv3DTranspose", "DepthwiseConv1D", "DepthwiseConv2D", "SeparableConv1D", "SeparableConv2D", 
           "Dense", "EinsumDense", "Embedding", "Identity", "Input", "InputLayer", "Lambda", "Masking", "Wrapper", 
           "InputSpec", "Layer", "Add", "add", "Average", "average", "Concatenate", "concatenate", "Dot", "dot", 
           "Maximum", "maximum", "Minimum", "minimum", "Multiply", "multiply", "Subtract", "subtract", 
           "BatchNormalization", "GroupNormalization", "LayerNormalization", "SpectralNormalization", "UnitNormalization", 
           "AveragePooling1D", "AveragePooling2D", "AveragePooling3D", "GlobalAveragePooling1D", "GlobalAveragePooling2D", 
           "GlobalAveragePooling3D", "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D", "MaxPooling1D", 
           "MaxPooling2D", "MaxPooling3D", "CategoryEncoding", "Discretization", "HashedCrossing", "Hashing", "AugMix", 
           "AutoContrast", "CenterCrop", "Equalization", "MaxNumBoundingBoxes", "MixUp", "RandAugment", "RandomBrightness", 
           "RandomColorDegeneration", "RandomColorJitter", "RandomContrast", "RandomCrop", "RandomFlip", "RandomGrayscale", 
           "RandomHue", "RandomPosterization", "RandomRotation", "RandomSaturation", "RandomSharpness", "RandomShear", 
           "RandomTranslation", "RandomZoom", "Resizing", "Solarization", "IndexLookup", "IntegerLookup", "MelSpectrogram", 
           "Normalization", "Pipeline", "Rescaling", "STFTSpectrogram", "StringLookup", "TextVectorization", 
           "ActivityRegularization", "AlphaDropout", "Dropout", "GaussianDropout", "GaussianNoise", "SpatialDropout1D", 
           "SpatialDropout2D", "SpatialDropout3D", "Cropping1D", "Cropping2D", "Cropping3D", "Flatten", "Permute",
              "RepeatVector", "Reshape", "UpSampling1D", "UpSampling2D", "UpSampling3D", "ZeroPadding1D", "ZeroPadding2D",
                "ZeroPadding3D", "Bidirectional", "ConvLSTM1D", "ConvLSTM2D", "ConvLSTM3D", "GRU", "GRUCell", "LSTM", "LSTMCell",
                    "RNN", "SimpleRNN", "SimpleRNNCell", "StackedRNNCells", "TimeDistributed", "serialize", "deserialize", "AbstractRNNCell"]
