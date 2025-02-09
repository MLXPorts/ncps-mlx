# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import mlx.core as np
import mlx.nn as nn
import mlx.optimizers as mx
import ncps.mini_keras
from ncps import wirings
from ncps.mlx import CfC, WiredCfCCell, LTC

# Simple training loop for MLX
def train_model(model, x, y, num_epochs=10, learning_rate=0.005):
    optimizer = mx.optimizers.Adam(learning_rate=learning_rate)
    
    def loss_fn(model, x, y):
        y_pred = model(x)
        return nn.losses.mse_loss(y_pred, y)
    
    for epoch in range(num_epochs):
        loss = loss_fn(model, x, y)
        grads = mx.grad(loss_fn)(model, x, y)
        optimizer.update(model, grads)
        print(f"Epoch {epoch+1}, Loss: {loss}")


in_features = 2
out_features = 1
N = 48  # Length of the time-series
# Input feature is a sine and a cosine wave
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
data_x = np.array(data_x)
data_y = np.array(data_y)
print("data_y.shape: ", str(data_y.shape))

# Define the wiring and model using the simplified API
wiring = wirings.AutoNCP(8, 1)  # Ensure wiring object is properly initialized
model = ncps.mini_keras.models.Sequential(
    [
        ncps.mini_keras.layers.InputLayer(input_shape=(None, 2)),
        # here we could potentially add layers before and after the LTC network
        LTC(wiring, return_sequences=True),
    ]
)
model.compile(
    optimizer=ncps.mini_keras.optimizers.Adam(0.01), loss='mean_squared_error'
)

model.summary()

# Example usage of the CfC model with PyTorch
for model in [
    CfC(in_features=in_features, hidden_size=32, out_features=out_features),
    WiredCfCCell(
        in_features=in_features, wiring=ncps.wirings.FullyConnected(8, out_features)
    ),
    WiredCfCCell(
        in_features=in_features,
        wiring=ncps.wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
    ),
    CfC(
        in_features=in_features,
        hidden_size=32,
        out_features=out_features,
        use_mm_rnn=True,
    ),
    WiredCfCCell(
        in_features=in_features,
        wiring=ncps.wirings.FullyConnected(8, out_features),
        use_mm_rnn=True,
    ),
    WiredCfCCell(
        in_features=in_features,
        wiring=ncps.wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
        use_mm_rnn=True,
    ),
]:
    train_model(model, data_x, data_y, num_epochs=10, learning_rate=0.01)
