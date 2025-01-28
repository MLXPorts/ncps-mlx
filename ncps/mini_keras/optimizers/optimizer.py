from ncps.mini_keras import backend
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.optimizers import base_optimizer

if backend.backend() == "tensorflow":
    from ncps.mini_keras.backend.tensorflow.optimizer import (
        TFOptimizer as BackendOptimizer,
    )
elif backend.backend() == "torch":
    from ncps.mini_keras.backend.torch.optimizers import (
        TorchOptimizer as BackendOptimizer,
    )
elif backend.backend() == "jax":
    from ncps.mini_keras.backend.jax.optimizer import JaxOptimizer as BackendOptimizer
elif backend.backend() == "mlx":
    from ncps.mini_keras.backend.mlx.optimizer import MLXOptimizer as BackendOptimizer
else:
    class BackendOptimizer(base_optimizer.BaseOptimizer):
        pass


@keras_mini_export(["ncps.mini_keras.Optimizer", "ncps.mini_keras.optimizers.Optimizer"])
class Optimizer(BackendOptimizer, base_optimizer.BaseOptimizer):
    pass


Optimizer.__doc__ = base_optimizer.BaseOptimizer.__doc__
base_optimizer_keyword_args = base_optimizer.base_optimizer_keyword_args
