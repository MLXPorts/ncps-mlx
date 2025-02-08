from ncps.mini_keras.backend import config

if config.backend() == "mlx":
    from ncps.mini_keras.backend.mlx.numpy import *
else:
    from ncps.mini_keras.backend.numpy.numpy import *
