import mlx.core as mx
import mlx.nn as nn
import numpy as np
import ncps
from ncps.mini_keras.models import Sequential
from ncps.mini_keras.layers import InputLayer, RNN, Dense
from ncps.mlx import WiredCfCCell  # Changed from CfCCell to WiredCfCCell
from ncps.wirings.wirings import NCP

# Generate synthetic time-series data
def generate_sine_data(seq_length=100, num_samples=1024):
    time = np.linspace(0, 2*np.pi, seq_length)
    x = np.sin(time).reshape(1, -1, 1).repeat(num_samples, axis=0)
    y = np.cos(time).reshape(1, -1, 1).repeat(num_samples, axis=0)
    return mx.array(x, dtype=mx.float32), mx.array(y, dtype=mx.float32)

# Create NCP wiring diagram
wiring = NCP(
    inter_neurons=16,  # Number of interconnecting neurons
    command_neurons=8, # Context-holding neurons
    motor_neurons=1,   # Output dimension
    sensory_fanout=4,  # Connections from input
    inter_fanout=4,    # Connections from inter to command neurons
    motor_fanin=2,     # Connections to output
    recurrent_command_synapses=3 # Feedback connections
)

# Ensure wiring is a valid Wiring instance
if not isinstance(wiring, ncps.wirings.Wiring):
    raise ValueError("The wiring argument must be a valid Wiring instance")

# Build CfC-based model
model = Sequential([
    InputLayer(input_shape=(None, 1)),
    RNN(WiredCfCCell(wiring=wiring), return_sequences=True),
    Dense(1)
])

# Training parameters
optimizer = nn.optimizers.Adam(learning_rate=3e-4)
loss_fn = nn.losses.mse

# Training loop
def train_step(x, y):
    def loss_fn(model, x, y):
        return mx.mean(mx.square(model(x) - y))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    return loss

# Generate and train on synthetic data
x_train, y_train = generate_sine_data()
for epoch in range(10):
    loss = train_step(x_train, y_train)
    mx.eval(loss)
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# Test inference
test_input = mx.array(np.sin(np.linspace(0, 2*np.pi, 100))[None,:,None], dtype=mx.float32)
prediction = model(test_input)
print("Output shape:", prediction.shape)  # Should be (1, 100, 1)
