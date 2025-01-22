import mlx.core as np
from absl.testing import parameterized

from ncps.mini_keras import backend
from ncps.mini_keras import testing
from ncps.mini_keras.utils import backend_utils


class BackendUtilsTest(testing.TestCase):
    @parameterized.named_parameters(
        ("numpy", "numpy"),
        ("jax", "jax"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
    )
    def test_dynamic_backend(self, name):
        dynamic_backend = backend_utils.DynamicBackend()
        x = np.random.uniform(size=[1, 2, 3]).astype("float32")

        if name == "numpy":
            dynamic_backend.set_backend(name)
            if backend.backend() != "numpy":
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "Currently, we cannot dynamically import the numpy backend",
                ):
                    y = dynamic_backend.numpy.log10(x)
            else:
                y = dynamic_backend.numpy.log10(x)
                self.assertIsInstance(y, np.ndarray)
        elif name == "tensorflow":
            import tensorflow as tf

            dynamic_backend.set_backend(name)
            y = dynamic_backend.numpy.log10(x)
            self.assertIsInstance(y, tf.Tensor)
        elif name == "torch":
            import torch

            dynamic_backend.set_backend(name)
            y = dynamic_backend.numpy.log10(x)
            self.assertIsInstance(y, torch.Tensor)

    def test_dynamic_backend_invalid_name(self):
        dynamic_backend = backend_utils.DynamicBackend()
        with self.assertRaisesRegex(ValueError, "Available backends are"):
            dynamic_backend.set_backend("abc")
