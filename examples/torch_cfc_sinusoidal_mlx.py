# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from ncps.mlx import CfC
from ncps.wirings import FullyConnected, NCP

def train_model(model, x, y, num_epochs=10, learning_rate=0.005):
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    def loss_fn(model, x, y):
        y_pred = model(x)
        return nn.losses.mse_loss(y_pred, y)
    
    for epoch in range(num_epochs):
        loss_value = loss_fn(model, x, y)
        grads = mx.grad(loss_fn)(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters())  # Ensure updates are applied
        print(f"Epoch {epoch+1}, Loss: {float(loss_value)}")

# Generate data
N = 48
in_features = 2
out_features = 1

data_x = mx.stack([
    mx.sin(mx.linspace(0, 3 * mx.pi, N)), 
    mx.cos(mx.linspace(0, 3 * mx.pi, N))
], axis=1)
data_x = mx.expand_dims(data_x, axis=0)
data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N)).reshape(1, N, 1)

# Initialize with explicit random seed for reproducibility
mx.random.seed(42)

# Test different model configurations
models = [
    CfC(
        32,  # units
        input_dim=in_features,
        output_dim=out_features,  # Add missing output dimension
        return_sequences=True
    ),
    CfC(
        8,  # units
        input_dim=in_features,
        output_dim=out_features,
        return_sequences=True,
        wiring=FullyConnected(8, out_features)
    ),
    CfC(
        24,  # total neurons (16 + 8)
        input_dim=in_features,
        output_dim=out_features,
        return_sequences=True,
        wiring=NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        )
    ),
]

for model in models:
    print(f"\nTraining {model.__class__.__name__}")
    train_model(model, data_x, data_y)
