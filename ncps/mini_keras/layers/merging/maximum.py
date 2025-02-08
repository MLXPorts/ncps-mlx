from ncps.mini_keras import ops
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.merging.base_merge import Merge


@keras_mini_export("ncps.mini_keras.layers.Maximum")
class Maximum(Merge):
    """Computes element-wise maximum on a list of inputs.

    It takes as input a list of tensors, all of the same shape,
    and returns a single tensor (also of the same shape).

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = ncps.mini_keras.layers.Maximum()([x1, x2])

    Usage in a Keras model:

    >>> input1 = ncps.mini_keras.layers.Input(shape=(16,))
    >>> x1 = ncps.mini_keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = ncps.mini_keras.layers.Input(shape=(32,))
    >>> x2 = ncps.mini_keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `y = ncps.mini_keras.layers.maximum([x1, x2])`
    >>> y = ncps.mini_keras.layers.Maximum()([x1, x2])
    >>> out = ncps.mini_keras.layers.Dense(4)(y)
    >>> model = ncps.mini_keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def _merge_function(self, inputs):
        return self._apply_merge_op_and_or_mask(ops.maximum, inputs)


@keras_mini_export("ncps.mini_keras.layers.maximum")
def maximum(inputs, **kwargs):
    """Functional interface to the `keras.layers.Maximum` layer.

    Args:
        inputs: A list of input tensors , all of the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the element-wise product of the inputs with the same
        shape as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = np.random.rand(*input_shape)
    >>> x2 = np.random.rand(*input_shape)
    >>> y = ncps.mini_keras.layers.maximum([x1, x2])

    Usage in a Keras model:

    >>> input1 = ncps.mini_keras.layers.Input(shape=(16,))
    >>> x1 = ncps.mini_keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = ncps.mini_keras.layers.Input(shape=(32,))
    >>> x2 = ncps.mini_keras.layers.Dense(8, activation='relu')(input2)
    >>> y = ncps.mini_keras.layers.maximum([x1, x2])
    >>> out = ncps.mini_keras.layers.Dense(4)(y)
    >>> model = ncps.mini_keras.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Maximum(**kwargs)(inputs)
