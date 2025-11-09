"""Example demonstrating enhanced training with different cell types."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from ncps.neurons import CTRNN, CTGRU, ELTC
from ncps.neurons.training import EnhancedLTCTrainer, TrainingConfig

# Generate sequence data
N = 1000  # sequence length
in_features = 2  # input dimension
out_features = 1  # output dimension
batch_size = 1  # using batch_size=1 for this example

# Generate input sequence: [batch_size, seq_length, features]
data_x = mx.stack([
    mx.sin(mx.linspace(0, 3 * mx.pi, N)), 
    mx.cos(mx.linspace(0, 3 * mx.pi, N))
], axis=1)  # Shape: [N, 2]
data_x = mx.expand_dims(data_x, axis=0)  # Shape: [1, N, 2]

# Generate target sequence: [batch_size, seq_length, output_dim]
data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N))  # Shape: [N]
data_y = mx.reshape(data_y, (1, N, out_features))  # Shape: [1, N, 1]

# Generate time delta information: [batch_size, seq_length]
# Use moderate time steps for stability
time_delta = mx.clip(mx.random.uniform(low=0.05, high=0.15, shape=(batch_size, N)), 0.05, 0.15)

# Initialize with explicit random seed for reproducibility
mx.random.seed(42)

# Training configuration with more conservative hyperparameters
config = TrainingConfig(
    target_accuracy=95.0,
    max_epochs=2000,  # Increased max epochs for slower learning
    patience=200,     # Increased patience
    weight_clip=0.1,  # Reduced weight clip
    bias_clip=0.05,   # Reduced bias clip
    max_grad_norm=0.5,  # Reduced grad norm
    learning_rate=0.001,  # Reduced learning rate
    min_learning_rate=0.0001,  # Lower min learning rate
    warmup_epochs=500,  # Longer warmup period
    noise_scale=0.01,  # Reduced noise scale
    noise_decay=0.9999,  # Slower noise decay
    momentum=0.99,  # Higher momentum
    grad_momentum=0.1  # Low gradient momentum
)

# Create trainer
trainer = EnhancedLTCTrainer(config)

# Define model configurations
model_configs = [
    (
        "CTRNN",
        CTRNN(
            units=32,  # Moderate number of units
            activation="tanh",  # Using string activation name
            cell_clip=0.5  # Moderate cell clip
        )
    ),
    (
        "CTGRU",
        CTGRU(
            units=32,  # Moderate number of units
            cell_clip=0.5  # Moderate cell clip
        )
    ),
    (
        "ELTC",
        ELTC(
            input_size=in_features,
            hidden_size=32,
            ode_unfolds=6,
            activation="tanh",
            cell_clip=0.5,  # Moderate cell clip
            return_sequences=True
        )
    )
]

# Train each model configuration
for name, model in model_configs:
    print(f"\nTraining {name}")
    
    # Initialize model by doing a forward pass
    dummy_input = mx.zeros((batch_size, 1, in_features))
    _ = model(dummy_input)
    
    # Print model parameters
    print("Model parameters:")
    def print_params(params, prefix=""):
        for name, param in params.items():
            if isinstance(param, dict):
                print(f"{prefix}{name}:")
                print_params(param, prefix + "  ")
            elif isinstance(param, list):
                print(f"{prefix}{name}: list of {len(param)} items")
                for i, item in enumerate(param):
                    if hasattr(item, 'shape'):
                        print(f"{prefix}  [{i}]: shape={item.shape}")
                    else:
                        print(f"{prefix}  [{i}]: {type(item)}")
            elif hasattr(param, 'shape'):
                print(f"{prefix}{name}: shape={param.shape}")
            else:
                print(f"{prefix}{name}: {type(param)}")
    
    print_params(model.parameters())
    
    # Train model
    history = trainer.train_with_accuracy_target(
        model=model,
        data_x=data_x,
        data_y=data_y,
        verbose=True
    )
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Best accuracy: {history['best_accuracy']:.2f}%")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Converged: {history['converged']}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    
    # Evaluate final predictions
    predictions = model(data_x, time_delta=time_delta)
    final_loss = mx.mean((predictions - data_y) ** 2)
    mx.eval(final_loss)  # Force evaluation
    print(f"Evaluation loss: {final_loss.item():.6f}")

print("\nTraining complete.")