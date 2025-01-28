
from ncps.mini_keras.api_export import keras_mini_export
from ncps.mini_keras.backend import random
from ncps.mini_keras.initializers.initializer import Initializer
from ncps.mini_keras.saving import serialization_lib
from ncps.mini_keras.backend.mlx import random_initializers as backend


class RandomInitializer(Initializer):
    def __init__(self, seed=None):
        self._init_seed = seed
        if seed is None:
            seed = random.make_default_seed()
        elif isinstance(seed, dict):
            seed = serialization_lib.deserialize_keras_object(seed)
        elif not isinstance(seed, (int, random.SeedGenerator)):
            raise ValueError(
                "`seed` argument should be an instance of "
                "`keras.random.SeedGenerator()` or an integer. "
                f"Received: seed={seed}"
            )
        self.seed = seed

    def get_config(self):
        seed_config = serialization_lib.serialize_keras_object(self._init_seed)
        return {"seed": seed_config}


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.RandomNormal",
        "ncps.mini_keras.initializers.random_normal",
    ]
)
class RandomNormal(RandomInitializer):
    """Random normal initializer."""

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return backend.random_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"mean": self.mean, "stddev": self.stddev}
        return {**base_config, **config}


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.TruncatedNormal",
        "ncps.mini_keras.initializers.truncated_normal",
    ]
)
class TruncatedNormal(RandomInitializer):
    """Initializer that generates a truncated normal distribution."""

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return backend.truncated_normal(
            shape=shape,
            mean=self.mean,
            stddev=self.stddev,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"mean": self.mean, "stddev": self.stddev}
        return {**base_config, **config}


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.RandomUniform",
        "ncps.mini_keras.initializers.random_uniform",
    ]
)
class RandomUniform(RandomInitializer):
    """Random uniform initializer."""

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return backend.random_uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"minval": self.minval, "maxval": self.maxval}
        return {**base_config, **config}


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.VarianceScaling",
        "ncps.mini_keras.initializers.variance_scaling",
    ]
)
class VarianceScaling(RandomInitializer):
    """Initializer that adapts its scale to the shape of its input tensors."""

    def __init__(
        self,
        scale=1.0,
        mode="fan_in",
        distribution="truncated_normal",
        seed=None,
    ):
        if scale <= 0.0:
            raise ValueError(
                "Argument `scale` must be positive float. "
                f"Received: scale={scale}"
            )
        allowed_modes = {"fan_in", "fan_out", "fan_avg"}
        if mode not in allowed_modes:
            raise ValueError(
                f"Invalid `mode` argument: {mode}. "
                f"Please use one of {allowed_modes}"
            )
        distribution = distribution.lower()
        if distribution == "normal":
            distribution = "truncated_normal"
        allowed_distributions = {
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }
        if distribution not in allowed_distributions:
            raise ValueError(
                f"Invalid `distribution` argument: {distribution}."
                f"Please use one of {allowed_distributions}"
            )
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return backend.variance_scaling(
            shape=shape,
            scale=self.scale,
            mode=self.mode,
            distribution=self.distribution,
            seed=self.seed,
            dtype=dtype,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "scale": self.scale,
            "mode": self.mode,
            "distribution": self.distribution,
        }
        return {**base_config, **config}


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.GlorotUniform",
        "ncps.mini_keras.initializers.glorot_uniform",
    ]
)
class GlorotUniform(VarianceScaling):
    """The Glorot uniform initializer, also called Xavier uniform initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.GlorotNormal",
        "ncps.mini_keras.initializers.glorot_normal",
    ]
)
class GlorotNormal(VarianceScaling):
    """The Glorot normal initializer, also called Xavier normal initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed,
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.LecunNormal",
        "ncps.mini_keras.initializers.lecun_normal",
    ]
)
class LecunNormal(VarianceScaling):
    """Lecun normal initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.LecunUniform",
        "ncps.mini_keras.initializers.lecun_uniform",
    ]
)
class LecunUniform(VarianceScaling):
    """Lecun uniform initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=1.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(["ncps.mini_keras.initializers.HeNormal", "ncps.mini_keras.initializers.he_normal"])
class HeNormal(VarianceScaling):
    """He normal initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="truncated_normal", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(["ncps.mini_keras.initializers.HeUniform", "ncps.mini_keras.initializers.he_uniform"])
class HeUniform(VarianceScaling):
    """He uniform variance scaling initializer."""

    def __init__(self, seed=None):
        super().__init__(
            scale=2.0, mode="fan_in", distribution="uniform", seed=seed
        )

    def get_config(self):
        return {
            "seed": serialization_lib.serialize_keras_object(self._init_seed)
        }


@keras_mini_export(
    [
        "ncps.mini_keras.initializers.Orthogonal",
        "ncps.mini_keras.initializers.orthogonal",
        "ncps.mini_keras.initializers.OrthogonalInitializer",
    ]
)
class Orthogonal(RandomInitializer):
    """Initializer that generates an orthogonal matrix."""

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        super().__init__(seed=seed)

    def __call__(self, shape, dtype=None):
        return backend.orthogonal(shape, gain=self.gain, seed=self.seed, dtype=dtype)

    def get_config(self):
        base_config = super().get_config()
        config = {"gain": self.gain}
        return {**base_config, **config}
