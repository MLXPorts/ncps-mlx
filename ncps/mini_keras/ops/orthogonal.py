from ncps.mini_keras import backend
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.backend import KerasTensor
from ncps.mini_keras.backend import any_symbolic_tensors
from ncps.mini_keras.ops.operation import Operation

class Orthogonal(Operation):
    def __init__(self, shape, gain=1.0):
        super().__init__()
        self.shape = shape
        self.gain = gain

    def call(self):
        return backend.mlx.orthogonal(self.shape, self.gain)

    def compute_output_spec(self):
        return KerasTensor(self.shape, dtype=backend.floatx())


@keras_mini_export(["ncps.mini_keras.ops.orthogonal", "ncps.mini_keras.ops.orthogonal"])
def orthogonal(shape, gain=1.0):
    """Orthogonal matrix initialization."""
    if any_symbolic_tensors((shape,)):
        return Orthogonal(shape, gain).symbolic_call()
    return backend.mlx.orthogonal(shape, gain)
