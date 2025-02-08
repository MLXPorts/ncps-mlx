from ncps.mini_keras import ops
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.input_spec import InputSpec
from ncps.mini_keras.layers.layer import Layer


@keras_mini_export("ncps.mini_keras.layers.RepeatVector")
class RepeatVector(Layer):
    """Repeats the input n times.

    Example:

    >>> x = ncps.mini_keras.Input(shape=(32,))
    >>> y = ncps.mini_keras.layers.RepeatVector(3)(x)
    >>> y.shape
    (None, 3, 32)

    Args:
        n: Integer, repetition factor.

    Input shape:
        2D tensor with shape `(batch_size, features)`.

    Output shape:
        3D tensor with shape `(batch_size, n, features)`.
    """

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        if not isinstance(n, int):
            raise TypeError(
                f"Expected an integer value for `n`, got {type(n)}."
            )
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        reshaped = ops.reshape(inputs, (input_shape[0], 1, input_shape[1]))
        return ops.repeat(reshaped, self.n, axis=1)

    def get_config(self):
        config = {"n": self.n}
        base_config = super().get_config()
        return {**base_config, **config}
