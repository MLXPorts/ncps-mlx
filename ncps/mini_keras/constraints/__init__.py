import inspect

from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.constraints.constraints import Constraint
from ncps.mini_keras.constraints.constraints import MaxNorm
from ncps.mini_keras.constraints.constraints import MinMaxNorm
from ncps.mini_keras.constraints.constraints import NonNeg
from ncps.mini_keras.constraints.constraints import UnitNorm
from ncps.mini_keras.saving import serialization_lib
from ncps.mini_keras.utils.naming import to_snake_case

ALL_OBJECTS = {
    Constraint,
    MaxNorm,
    MinMaxNorm,
    NonNeg,
    UnitNorm,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)


@keras_mini_export("ncps.mini_keras.constraints.serialize")
def serialize(constraint):
    return serialization_lib.serialize_keras_object(constraint)


@keras_mini_export("ncps.mini_keras.constraints.deserialize")
def deserialize(config, custom_objects=None):
    """Return a Keras constraint object via its config."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_mini_export("ncps.mini_keras.constraints.get")
def get(identifier):
    """Retrieve a Keras constraint object via an identifier."""
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
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret constraint identifier: {identifier}"
        )
