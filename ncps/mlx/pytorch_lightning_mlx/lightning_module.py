from typing import Any, Optional, Callable


class LightningModule:
    """
    Minimal LightningModule replacement using MLX.
    Handles model definitions, forward passes, and training/validation logic.
    """

    def __init__(self):
        self.params = []

    def forward(self, *args, **kwargs) -> Any:
        """Override to define forward pass."""
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        """Override to define training step."""
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[Any]:
        """Override to define validation step."""
        pass

    def configure_optimizers(self):
        """Override to configure optimizers."""
        raise NotImplementedError

    def parameters(self):
        """Returns all parameters of the model."""
        return self.params


class MLXOptimizer:
    """
    Replacement for PyTorch optimizer using MLX.
    """

    def __init__(self, params, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        """Performs optimization step."""
        for param in self.params:
            grad = param.grad
            if grad is not None:
                param.data -= self.lr * grad

    def zero_grad(self):
        """Clears gradients."""
        for param in self.params:
            param.grad = None


def rank_zero_only(fn: Callable) -> Callable:
    """Ensures the function runs only on the rank-zero process in distributed mode."""
    def wrapped_fn(*args, **kwargs):
        # Assume single-threaded for now (non-distributed)
        return fn(*args, **kwargs)

    return wrapped_fn


@rank_zero_only
def rank_zero_print(*args):
    print(*args)
