import mlx.core as mx
import mlx.nn as nn
from ncps import wirings
from ncps.mlx import CfC, WiredCfCCell

# Data preparation using MLX ops
N = 48  # Length of the time-series
in_features = 2
out_features = 1

# Input feature is a sine and a cosine wave using MLX ops
t = mx.linspace(0, 3 * mx.pi, N)
sin_wave = mx.sin(t)
cos_wave = mx.cos(t)
data_x = mx.stack([sin_wave, cos_wave], axis=1)
data_x = mx.expand_dims(data_x, axis=0)  # Add batch dimension

# Target output is a sine with double the frequency
t_out = mx.linspace(0, 6 * mx.pi, N)  
data_y = mx.sin(t_out)
data_y = mx.reshape(data_y, [1, N, 1])

# List of model configurations
model_configs = [
    CfC(units=32),
    WiredCfCCell(
        wiring=wirings.FullyConnected(8, out_features)
    ),
    WiredCfCCell(
        wiring=wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        )
    ),
    CfC(
        units=32,
        mode="pure"
    ),
    WiredCfCCell(
        wiring=wirings.FullyConnected(8, out_features),
        mode="pure"
    ),
    WiredCfCCell(
        wiring=wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
        mode="pure"
    ),
]

# Training using MLX
class Model(nn.Module):
    def __init__(self, rnn_cell):
        super().__init__()
        self.rnn = rnn_cell
        
    def __call__(self, x, training=True):
        # Process sequence
        h = None
        outputs = []
        for t in range(x.shape[1]):
            y, h = self.rnn(x[:, t:t+1, :], h)
            outputs.append(y)
        return mx.concatenate(outputs, axis=1)

# Training parameters
optimizer = nn.optimizers.Adam(learning_rate=0.01)

def train_step(model, x, y):
    def loss_fn(model, x, y):
        return mx.mean(mx.square(model(x) - y))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    return loss

# Train each model configuration
for rnn_cell in model_configs:
    model = Model(rnn_cell)
    print(f"\nTraining model: {rnn_cell.__class__.__name__}")
    
    # Training loop
    for epoch in range(10):
        loss = train_step(model, data_x, data_y)
        mx.eval(loss)
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    
    # Evaluate
    with mx.eval_mode():
        final_loss = mx.mean(mx.square(model(data_x) - data_y))
        print(f"\nFinal MSE for {rnn_cell.__class__.__name__}: {final_loss.item():.4f}")
