import inspect

import mlx.core as np

from ncps.mini_keras import backend
from ncps.mini_keras import ops
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.initializers.constant_initializers import STFT
from ncps.mini_keras.initializers.constant_initializers import Constant
from ncps.mini_keras.initializers.constant_initializers import Identity
from ncps.mini_keras.initializers.constant_initializers import Ones
from ncps.mini_keras.initializers.constant_initializers import Zeros
from ncps.mini_keras.initializers.initializer import Initializer
from ncps.mini_keras.initializers.random_initializers import GlorotNormal
from ncps.mini_keras.initializers.random_initializers import GlorotUniform
from ncps.mini_keras.initializers.random_initializers import HeNormal
from ncps.mini_keras.initializers.random_initializers import HeUniform
from ncps.mini_keras.initializers.random_initializers import LecunNormal
from ncps.mini_keras.initializers.random_initializers import LecunUniform
from ncps.mini_keras.initializers.random_initializers import Orthogonal
from ncps.mini_keras.initializers.random_initializers import RandomNormal
from ncps.mini_keras.initializers.random_initializers import RandomUniform
from ncps.mini_keras.initializers.random_initializers import TruncatedNormal
from ncps.mini_keras.initializers.random_initializers import VarianceScaling
from ncps.mini_keras.saving import serialization_lib
from ncps.mini_keras.utils.naming import to_snake_case

ALL_OBJECTS = {
    Initializer,
    Constant,
    Identity,
    Ones,
    STFT,
    Zeros,
    GlorotNormal,
    GlorotUniform,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    Orthogonal,
    RandomNormal,
    RandomUniform,
    TruncatedNormal,
    VarianceScaling,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}
ALL_OBJECTS_DICT.update(
    {to_snake_case(cls.__name__): cls for cls in ALL_OBJECTS}
)
# Aliases
ALL_OBJECTS_DICT.update(
    {
        "IdentityInitializer": Identity,  # For compatibility
        "normal": RandomNormal,
        "one": Ones,
        "STFTInitializer": STFT,  # For compatibility
        "OrthogonalInitializer": Orthogonal,  # For compatibility
        "uniform": RandomUniform,
        "zero": Zeros,
    }
)


@keras_mini_export("ncps.mini_keras.initializers.serialize")
def serialize(initializer):
    """Returns the initializer configuration as a Python dict."""
    return serialization_lib.serialize_keras_object(initializer)


@keras_mini_export("ncps.mini_keras.initializers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras initializer object via its configuration."""
    return serialization_lib.deserialize_keras_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@keras_mini_export("ncps.mini_keras.initializers.get")
def get(identifier):
    """Retrieves a Keras initializer object via an identifier.

    The `identifier` may be the string name of a initializers function or class
    (case-sensitively).

    >>> identifier = 'Ones'
    >>> keras.initializers.get(identifier)
    <...keras.initializers.initializers.Ones...>

    You can also specify `config` of the initializer to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Initializer` class.

    >>> cfg = {'class_name': 'Ones', 'config': {}}
    >>> keras.initializers.get(cfg)
    <...keras.initializers.initializers.Ones...>

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    You may also pass a callable function with a signature that includes `shape`
    and `dtype=None` as an identifier.

    >>> fn = lambda shape, dtype=None: ops.ones(shape, dtype)
    >>> keras.initializers.get(fn)
    <function <lambda> at ...>

    Alternatively, you can pass a backend tensor or numpy array as the
    `identifier` to define the initializer values directly. Note that when
    calling the initializer, the specified `shape` argument must be the same as
    the shape of the tensor.

    >>> tensor = ops.ones(shape=(5, 5))
    >>> keras.initializers.get(tensor)
    <function get.<locals>.initialize_fn at ...>

    Args:
        identifier: A string, dict, callable function, or tensor specifying
            the initializer. If a string, it should be the name of an
            initializer. If a dict, it should contain the configuration of an
            initializer. Callable functions or predefined tensors are also
            accepted.

    Returns:
        Initializer instance base on the input identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        config = {"class_name": str(identifier), "config": {}}
        obj = deserialize(config)
    elif ops.is_tensor(identifier) or isinstance(
        identifier, (np.generic, np.ndarray)
    ):

        def initialize_fn(shape, dtype=None):
            dtype = backend.standardize_dtype(dtype)
            if backend.standardize_shape(shape) != backend.standardize_shape(
                identifier.shape
            ):
                raise ValueError(
                    f"Expected `shape` to be {identifier.shape} for direct "
                    f"tensor as initializer. Received shape={shape}"
                )
            return ops.cast(identifier, dtype)

        obj = initialize_fn
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret initializer identifier: {identifier}"
        )
