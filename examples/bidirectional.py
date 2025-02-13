# Copyright (2017-2020)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import numpy as np
import mlx.core as mx
import mlx.nn as nn
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

# Create bidirectional model using MLX
class BidirectionalRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_rnn = LTC(auto_ncp, cell_class=EnhancedLTCCell, solver="rk4", ode_unfolds=6)
        self.backward_rnn = LTC(auto_ncp, cell_class=EnhancedLTCCell, solver="rk4", ode_unfolds=6)
        self.dense = nn.Linear(auto_ncp.units * 2, 1)  # *2 for bidirectional
    
    def __call__(self, x):
        # Forward pass
        h_forward = None
        forward_outputs = []
        for t in range(x.shape[1]):
            y, h_forward = self.forward_rnn(x[:, t:t+1, :], h_forward)
            forward_outputs.append(y)
        
        # Backward pass
        h_backward = None
        backward_outputs = []
        for t in range(x.shape[1]-1, -1, -1):
            y, h_backward = self.backward_rnn(x[:, t:t+1, :], h_backward)
            backward_outputs.insert(0, y)
        
        # Concatenate forward and backward outputs
        forward_seq = mx.concatenate(forward_outputs, axis=1)
        backward_seq = mx.concatenate(backward_outputs, axis=1)
        combined = mx.concatenate([forward_seq, backward_seq], axis=-1)
        
        # Apply dense layer
        return self.dense(combined)

model = BidirectionalRNN()

# Training parameters
optimizer = nn.optimizers.Adam(learning_rate=0.01)

# Training loop
def train_step(model, x, y):
    def loss_fn(model, x, y):
        return mx.mean(mx.square(model(x) - y))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    return loss

# Train the model
for epoch in range(200):
    loss = train_step(model, data_x, data_y)
    mx.eval(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Test inference
with mx.eval_mode():
    prediction = model(data_x)
    final_loss = mx.mean(mx.square(prediction - data_y))
    print(f"\nFinal MSE: {final_loss.item():.4f}")
