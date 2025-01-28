from ncps.mini_keras.api_export import keras_mini_export

GLOBAL_CUSTOM_OBJECTS = {}

@keras_mini_export("ncps.mini_keras.utils.register_keras_serializable")
def register_keras_serializable(package="Custom", name=None):
    """Registers an object with the Keras serialization framework.

    This decorator injects the decorated class or function into Keras's global
    custom object dictionary so it can be serialized and deserialized
    without needing an entry in the user's custom object scope.

    Example:

    ```python
    @register_keras_serializable(package='MyFramework')
    class MyClass:
        def __init__(self, value):
            self.value = value

        def get_config(self):
            return {'value': self.value}

        @classmethod
        def from_config(cls, config):
            return cls(**config)
    ```

    Args:
        package: The package that this class belongs to. This is used for
            namespacing to ensure that there are no naming collisions between 
            objects from different libraries.
        name: The name to serialize this class under in the global object
            scope. If None, the class's name will be used.

    Returns:
        A decorator that registers the decorated class with the Keras custom
        object system.
    """
    def decorator(arg):
        class_name = name or arg.__name__
        registered_name = package + '>' + class_name

        GLOBAL_CUSTOM_OBJECTS[registered_name] = arg
        return arg

    return decorator

register_mini_keras_serializable = register_keras_serializable
