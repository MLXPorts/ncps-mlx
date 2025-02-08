from ncps.mini_keras import ops
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.merging.base_merge import Merge


@keras_mini_export("ncps.mini_keras.layers.Subtract")
class Subtract(Merge):
    """Performs elementwise subtraction.

    It takes as input a list of tensors of size 2 both of the
    same shape, and returns a single tensor (inputs[0] - inputs[1])
    of same shape.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = ncps.mini_keras.layers.Subtract()([x1, x2])

    Usage in a Keras model:

    >>> input1 = ncps.mini_keras.layers.Input(shape=(16,))
    >>> x1 = ncps.mini_keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = ncps.mini_keras.layers.Input(shape=(32,))
    >>> x2 = ncps.mini_keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `subtracted = ncps.mini_keras.layers.subtract([x1, x2])`
    >>> subtracted = ncps.mini_keras.layers.Subtract()([x1, x2])
    >>> out = ncps.mini_keras.layers.Dense(4)(subtracted)
    >>> model = ncps.mini_keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def build(self, input_shape):
        super().build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: input_shape={input_shape}"
            )

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                "A `Subtract` layer should be called on exactly 2 inputs. "
                f"Received: inputs={inputs}"
            )
        return ops.subtract(inputs[0], inputs[1])


@keras_mini_export("ncps.mini_keras.layers.subtract")
def subtract(inputs, **kwargs):
    """Functional interface to the `keras.layers.Subtract` layer.

    Args:
        inputs: A list of input tensors of size 2, each tensor of
            the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the difference of the inputs. It has the same shape
        as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = ncps.mini_keras.layers.subtract([x1, x2])

    Usage in a Keras model:

    >>> input1 = ncps.mini_keras.layers.Input(shape=(16,))
    >>> x1 = ncps.mini_keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = ncps.mini_keras.layers.Input(shape=(32,))
    >>> x2 = ncps.mini_keras.layers.Dense(8, activation='relu')(input2)
    >>> subtracted = ncps.mini_keras.layers.subtract([x1, x2])
    >>> out = ncps.mini_keras.layers.Dense(4)(subtracted)
    >>> model = ncps.mini_keras.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Subtract(**kwargs)(inputs)
