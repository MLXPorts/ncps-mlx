import inspect

from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.quantizers.quantizers import AbsMaxQuantizer
from ncps.mini_keras.quantizers.quantizers import Quantizer
from ncps.mini_keras.quantizers.quantizers import abs_max_quantize
from ncps.mini_keras.quantizers.quantizers import compute_float8_amax_history
from ncps.mini_keras.quantizers.quantizers import compute_float8_scale
from ncps.mini_keras.quantizers.quantizers import fake_quant_with_min_max_vars
from ncps.mini_keras.quantizers.quantizers import quantize_and_dequantize
from ncps.mini_keras.saving import serialization_lib
from ncps.mini_keras.utils.naming import to_snake_case

ALL_OBJECTS = {Quantizer, AbsMaxQuantizer}
ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)


@keras_mini_export("ncps.mini_keras.quantizers.serialize")
def serialize(initializer):
    return serialization_lib.serialize_keras_object(initializer)


@keras_mini_export("ncps.mini_keras.quantizers.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras quantizer object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_mini_export("ncps.mini_keras.quantizers.get")
def get(identifier, **kwargs):
    """Retrieve a Keras quantizer object via an identifier."""
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj(kwargs)
        return obj
    else:
        raise ValueError(
            f"Could not interpret quantizer identifier: {identifier}"
        )
