from ncps.mini_keras.api_export import keras_mini_export

# Unique source of truth for the version number.
__version__ = "3.8.0"


@keras_mini_export("ncps.mini_keras.version")
def version():
    return __version__
