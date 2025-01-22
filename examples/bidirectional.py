# Copyright (2017-2020)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import numpy as np
import mlx.core as mx
from ncps.mini_keras import models, layers
from ncps.mlx import LTC, EnhancedLTCCell
from ncps import wirings

def generate_data(N):
    # Generate data
    # Input feature is a sine and a cosine wave
    data_x = mx.stack(
        [mx.sin(mx.linspace(0, 3 * mx.pi, N)), mx.cos(mx.linspace(0, 3 * mx.pi, N))], axis=1
    )
    data_x = mx.expand_dims(data_x, axis=0).astype(mx.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N)).reshape([1, N, 1]).astype(mx.float32)
    print("data_y.shape: ", str(data_y.shape))

    data_x = mx.array(data_x)  # Convert to MLX array
    data_y = mx.array(data_y)
    return data_x, data_y

N = 48
data_x, data_y = generate_data(N)

# Use AutoNCP wiring
auto_ncp = wirings.AutoNCP(32, 1)  # 32 neurons, 1 output

# Create model using mini_keras bidirectional wrapper
model = models.Sequential([
    layers.InputLayer(input_shape=(None, 2)),
    layers.Bidirectional(
        layers.RNN(
            LTC(auto_ncp, cell_class=EnhancedLTCCell, solver="rk4", ode_unfolds=6),
            return_sequences=True
        )
    ),
    layers.Dense(1)
])

# Use mini_keras training
model.compile(
    optimizer='adam',
    loss='mse',
    learning_rate=0.01
)
model.fit(data_x, data_y, batch_size=1, epochs=200)
