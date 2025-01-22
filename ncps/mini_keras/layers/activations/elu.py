from ncps.mini_keras import activations
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.layers.layer import Layer


@keras_mini_export("ncps.mini_keras.layers.ELU")
class ELU(Layer):
    """Applies an Exponential Linear Unit function to an output.

    Formula:

    ```
    f(x) = alpha * (exp(x) - 1.) for x < 0
    f(x) = x for x >= 0
    ```

    Args:
        alpha: float, slope of negative section. Defaults to `1.0`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.supports_masking = True
        self.built = True

    def call(self, inputs):
        return activations.elu(inputs, alpha=self.alpha)

    def compute_output_shape(self, input_shape):
        return input_shape
