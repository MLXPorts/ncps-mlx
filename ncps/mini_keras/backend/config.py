import json
import os

from ncps.mini_keras.api_export import keras_mini_export

# The type of float to use throughout a session.
_FLOATX = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT = "channels_last"

# Default backend is MLX - this is a fork of Keras minimal for NCP usage
_BACKEND = "mlx"

# Mini-Keras home directory (separate from Keras to avoid conflicts)
_MINI_KERAS_DIR = os.path.join(os.path.expanduser("~"), ".mini_keras")

# Support both Keras and Mini-Keras config locations
if "KERAS_HOME" in os.environ:
    _KERAS_DIR = os.environ.get("KERAS_HOME")
elif "MINI_KERAS_HOME" in os.environ:
    _KERAS_DIR = os.environ.get("MINI_KERAS_HOME") 
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _KERAS_DIR = os.path.join(_keras_base_dir, ".keras")
    _MINI_KERAS_DIR = os.path.join(_keras_base_dir, ".mini_keras")

def keras_home():
    """Private accessor for Keras home location.
    Checks both Keras and Mini-Keras locations."""
    return _KERAS_DIR


@keras_mini_export(["ncps.mini_keras.config.floatx", "ncps.mini_keras.backend.floatx"])
def floatx():
    """Return the default float type, as a string.
    
    Note: This implementation uses mlx.core.array (a NumPy-compatible array) under the hood,
    so all NumPy dtypes and array operations are supported. MLX arrays are accelerated 
    versions of NumPy ndarrays.

    Returns:
        String, one of: 'float16', 'float32', 'float64'

    Example:

    >>> ncps.mini_keras.config.floatx()
    'float32'
    """
    return _FLOATX


@keras_mini_export(["ncps.mini_keras.config.set_floatx", "ncps.mini_keras.backend.set_floatx"])
def set_floatx(value):
    """Set the default float dtype.
    
    Note: MLX arrays inherit from NumPy's ndarray, so they support all NumPy float types
    while providing accelerated operations through the MLX backend.

    Args:
        value: String; 'float16', 'float32', or 'float64'. These correspond to
               NumPy-compatible dtypes that are accelerated by MLX.

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(
            f"Unknown `floatx` value: {value}. "
            f"Expected one of {accepted_dtypes}"
        )
    _FLOATX = str(value)


@keras_mini_export(["ncps.mini_keras.config.epsilon", "ncps.mini_keras.backend.epsilon"])
def epsilon():
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:

    >>> keras.config.epsilon()
    1e-07

    """
    return _EPSILON


@keras_mini_export(["ncps.mini_keras.config.set_epsilon", "ncps.mini_keras.backend.set_epsilon"])
def set_epsilon(value):
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Examples:
    >>> keras.config.epsilon()
    1e-07

    >>> keras.config.set_epsilon(1e-5)
    >>> keras.config.epsilon()
    1e-05

    >>> # Set it back to the default value.
    >>> keras.config.set_epsilon(1e-7)

    """
    global _EPSILON
    _EPSILON = value


@keras_mini_export(
    [
        "ncps.mini_keras.config.image_data_format",
        "ncps.mini_keras.backend.image_data_format",
    ]
)
def image_data_format():
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:

    >>> keras.config.image_data_format()
    'channels_last'

    """
    return _IMAGE_DATA_FORMAT


@keras_mini_export(
    [
        "ncps.mini_keras.config.set_image_data_format",
        "ncps.mini_keras.backend.set_image_data_format",
    ]
)
def set_image_data_format(data_format):
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Examples:

    >>> keras.config.image_data_format()
    'channels_last'

    >>> keras.config.set_image_data_format('channels_first')
    >>> keras.config.image_data_format()
    'channels_first'

    >>> # Set it back to `'channels_last'`
    >>> keras.config.set_image_data_format('channels_last')

    """
    global _IMAGE_DATA_FORMAT
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    _IMAGE_DATA_FORMAT = data_format


@keras_mini_export("ncps.mini_keras.config.enable_flash_attention")
def enable_flash_attention():
    """Enable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once enabled, supported layers like `MultiHeadAttention` will **attempt** to
    use flash attention for faster computations. By default, this feature is
    enabled.

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.
    """
    from ncps.mini_keras.backend.common import global_state

    global_state.set_global_attribute("flash_attention", None)


@keras_mini_export("ncps.mini_keras.config.disable_flash_attention")
def disable_flash_attention():
    """Disable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once disabled, supported layers like `MultiHeadAttention` will not
    use flash attention for faster computations.
    """
    from ncps.mini_keras.backend.common import global_state

    global_state.set_global_attribute("flash_attention", False)


@keras_mini_export("ncps.mini_keras.config.is_flash_attention_enabled")
def is_flash_attention_enabled():
    """Checks whether flash attention is globally enabled in Keras.

    Flash attention is a performance-optimized method for computing attention
    in large models, such as transformers, allowing for faster and more
    memory-efficient operations. This function checks the global Keras
    configuration to determine if flash attention is enabled for compatible
    layers (e.g., `MultiHeadAttention`).

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.

    Returns:
        `False` if disabled; otherwise, it indicates that it is enabled.
    """
    from ncps.mini_keras.backend.common import global_state

    return global_state.get_global_attribute("flash_attention", default=None)


def standardize_data_format(data_format):
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


# Config file handling
def _get_mini_keras_dir():
    if "MINI_KERAS_HOME" in os.environ:
        return os.environ.get("MINI_KERAS_HOME")
    return _MINI_KERAS_DIR


# Initialize config directory if needed
config_dir = _get_mini_keras_dir()
if not os.path.exists(config_dir):
    try:
        os.makedirs(config_dir)
    except OSError:
        # Handle permission denied and race conditions
        pass

# Try loading from Keras config first, fall back to mini-keras
_config_path = os.path.expanduser(os.path.join(_KERAS_DIR, "keras.json"))
_mini_config_path = os.path.expanduser(os.path.join(_MINI_KERAS_DIR, "mini_keras.json"))

if os.path.exists(_config_path):
    # Load from Keras config
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get("floatx", floatx())
    _epsilon = _config.get("epsilon", _EPSILON)
    _image_data_format = _config.get("image_data_format", _IMAGE_DATA_FORMAT)

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
elif os.path.exists(_mini_config_path):
    # Load from mini-keras config
    try:
        with open(_mini_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get("floatx", floatx())
    _epsilon = _config.get("epsilon", _EPSILON)
    _image_data_format = _config.get("image_data_format", _IMAGE_DATA_FORMAT)

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)

# Create config in mini-keras location if neither exists
if not os.path.exists(_config_path) and not os.path.exists(_mini_config_path):
    _config = {
        "floatx": floatx(),
        "epsilon": _EPSILON,
        "backend": _BACKEND,
        "image_data_format": _IMAGE_DATA_FORMAT
    }
    config_dir = _MINI_KERAS_DIR
    config_file = _mini_config_path
    try:
        with open(config_file, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Handle permission denied
        pass


@keras_mini_export(
    [
        "ncps.mini_keras.config.backend",
        "ncps.mini_keras.backend.backend",
    ]
)
def backend():
    """Returns the current backend name.

    Returns:
        String 'mlx', as this is an MLX-specific implementation.

    Note: While the original Keras supports multiple backends, this mini_keras
    implementation is specifically for MLX backend usage in Neural Circuit Policies.
    NumPy-style operations are supported but are converted to MLX arrays internally.
    """
    return _BACKEND
