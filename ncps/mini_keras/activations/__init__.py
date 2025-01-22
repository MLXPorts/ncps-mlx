"""Activation functions for mini-keras.

This module provides common activation functions used in neural networks.
"""

import types

from .activations import celu
from .activations import elu
from .activations import exponential
from .activations import gelu
from .activations import glu
from .activations import hard_shrink
from .activations import hard_sigmoid
from .activations import hard_silu
from .activations import hard_tanh
from .activations import leaky_relu
from .activations import linear
from .activations import log_sigmoid
from .activations import log_softmax
from .activations import mish
from .activations import relu
from .activations import relu6
from .activations import selu
from .activations import sigmoid
from .activations import silu
from .activations import soft_shrink
from .activations import softmax
from .activations import softplus
from .activations import softsign
from .activations import sparse_plus
from .activations import sparsemax
from .activations import squareplus
from .activations import lecun_tanh
from .activations import tanh
from .activations import tanh_shrink
from .activations import threshold
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.saving import object_registration
from ncps.mini_keras.saving import serialization_lib

ALL_OBJECTS = {
    relu,
    leaky_relu,
    relu6,
    softmax,
    celu,
    elu,
    selu,
    softplus,
    softsign,
    squareplus,
    soft_shrink,
    sparse_plus,
    silu,
    gelu,
    glu,
    tanh,
    lecun_tanh,
    tanh_shrink,
    threshold,
    sigmoid,
    exponential,
    hard_sigmoid,
    hard_silu,
    hard_tanh,
    hard_shrink,
    linear,
    mish,
    log_softmax,
    log_sigmoid,
    sparsemax,
}

ALL_OBJECTS_DICT = {fn.__name__: fn for fn in ALL_OBJECTS}
# Additional aliases
ALL_OBJECTS_DICT["swish"] = silu
ALL_OBJECTS_DICT["hard_swish"] = hard_silu


@keras_mini_export("ncps.mini_keras.activations.serialize")
def serialize(activation):
    fn_config = serialization_lib.serialize_keras_object(activation)
    if "config" not in fn_config:
        raise ValueError(
            f"Unknown activation function '{activation}' cannot be "
            "serialized due to invalid function name. Make sure to use "
            "an activation name that matches the references defined in "
            "activations.py or use "
            "`@keras.saving.register_keras_serializable()`"
            "to register any custom activations. "
            f"config={fn_config}"
        )
    if not isinstance(activation, types.FunctionType):
        # Case for additional custom activations represented by objects
        return fn_config
    if (
        isinstance(fn_config["config"], str)
        and fn_config["config"] not in globals()
    ):
        # Case for custom activation functions from external activations modules
        fn_config["config"] = object_registration.get_registered_name(
            activation
        )
        return fn_config
    # Case for keras.activations builtins (simply return name)
    return fn_config["config"]


@keras_mini_export("ncps.mini_keras.activations.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras activation function via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_mini_export("ncps.mini_keras.activations.get")
def get(identifier):
    """Retrieve a Keras activation function via an identifier."""
    if identifier is None:
        return linear
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier
    if callable(obj):
        return obj
    raise ValueError(
        f"Could not interpret activation function identifier: {identifier}"
    )
