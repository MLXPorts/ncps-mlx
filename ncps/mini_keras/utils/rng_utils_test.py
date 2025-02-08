try:
    import mlx.core as np
except ImportError:
    import numpy as np
    
import pytest
import tensorflow as tf

import keras
from ncps.mini_keras import backend
from ncps.mini_keras.testing import test_case
from ncps.mini_keras.utils import rng_utils


class TestRandomSeedSetting(test_case.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support random seed setting.",
    )
    def test_set_random_seed(self):
        def get_model_output():
            model = ncps.mini_keras.Sequential(
                [
                    ncps.mini_keras.layers.Dense(10),
                    ncps.mini_keras.layers.Dropout(0.5),
                    ncps.mini_keras.layers.Dense(10),
                ]
            )
            x = np.random.random((32, 10)).astype("float32")
            ds = tf.data.Dataset.from_tensor_slices(x).shuffle(32).batch(16)
            return model.predict(ds)

        rng_utils.set_random_seed(42)
        y1 = get_model_output()
        rng_utils.set_random_seed(42)
        y2 = get_model_output()
        self.assertAllClose(y1, y2)
