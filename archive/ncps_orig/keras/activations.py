"""Custom activation functions for Keras neural circuit implementations."""

import keras
from keras import ops, layers


@keras.saving.register_keras_serializable(package="ncps")
class LeCunTanh(layers.Layer):
    """LeCun improved tanh activation layer.
    
    This is an improved version of tanh that leads to faster training.
    See: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    """
    
    def call(self, x):
        """Apply LeCun tanh activation."""
        return 1.7159 * ops.tanh(0.666 * x)


# Create function wrapper for the layer
def lecun_tanh(x):
    """LeCun improved tanh activation function."""
    layer = LeCunTanh()
    return layer(x)


# Register custom activation with Keras
keras.utils.get_custom_objects()['LeCunTanh'] = LeCunTanh