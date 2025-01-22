import numpy as np
import mlx.core as mx
from ncps.mini_keras import models, layers
from ncps.mlx import LTC, EnhancedLTCCell
from ncps import wirings

def generate_data(N):
    in_features = 2
    out_features = 1
    # Input feature is a sine and a cosine wave
    data_x = mx.stack(
        [mx.sin(mx.linspace(0, 3 * mx.pi, N)), mx.cos(mx.linspace(0, 3 * mx.pi, N))], axis=1
    )
    data_x = mx.expand_dims(data_x, axis=0).astype(mx.float32)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N)).reshape([1, N, 1]).astype(mx.float32)
    return data_x, data_y

N = 48
data_x, data_y = generate_data(N)

# Create NCP wiring configuration
ncp_wiring = wirings.NCP(
    inter_neurons=20,
    command_neurons=10,
    motor_neurons=1,
    sensory_fanout=4,
    inter_fanout=5,
    recurrent_command_synapses=6,
    motor_fanin=4,
)

# Create model using mini_keras
model = models.Sequential([
    layers.InputLayer(input_shape=(None, 2)),
    layers.RNN(
        LTC(ncp_wiring, cell_class=EnhancedLTCCell, solver="rk4", ode_unfolds=6),
        return_sequences=True
    )
])

# Use mini_keras training
model.compile(
    optimizer='adam',
    loss='mse',
    learning_rate=0.01
)
model.fit(data_x, data_y, batch_size=1, epochs=400)
