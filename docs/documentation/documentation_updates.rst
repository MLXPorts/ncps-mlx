Documentation Updates Needed
============================

[Previous sections remain the same…]

9. Example Notebook Updates
---------------------------

9.1 MLX CfC Example (examples/notebooks/mlx_cfc_example.ipynb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current notebook needs updates to reflect the new API:

1. Model Creation Updates

.. code:: python

from ncps.mlx import CfC, CfCCell
from ncps.wirings import AutoNCP, NCP

# Create wiring
wiring = AutoNCP(units=32, output_size=1)

class BasicSequenceModel(nn.Module):
    """Basic sequence model using CfC."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cfc = CfC(
            cell=CfCCell(
                wiring=wiring,
                activation="tanh",
                backbone_units=[64],
                backbone_layers=1
            ),
            return_sequences=False
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

class BidirectionalSequenceModel(nn.Module):
    """Bidirectional sequence model using CfC."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cfc = CfC(
            cell=CfCCell(
                wiring=wiring,
                activation="tanh"
            ),
            bidirectional=True,
            merge_mode="concat",
            return_sequences=True
        )
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

class DeepSequenceModel(nn.Module):
    """Deep sequence model using stacked CfC layers."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cfc = CfC(
            cell=CfCCell(
                wiring=wiring,
                activation="tanh",
                backbone_units=[64, 64],
                backbone_layers=2
            ),
            return_sequences=True
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

2. Add State Management Examples

.. code:: python

def process_with_state(model, X, initial_state=None):
    """Process sequence with explicit state management."""
    if initial_state is None:
        batch_size = X.shape[0]
        initial_state = mx.zeros((batch_size, model.cfc.cell.units))

    outputs, final_state = model.cfc(X, initial_state=initial_state)
    return model.output_layer(outputs), final_state

3. Add Time-Aware Processing

.. code:: python

def generate_variable_time_data(batch_size, seq_length):
    """Generate data with variable time steps."""
    # Base data generation
    X, y = generate_base_data(batch_size, seq_length)

    # Generate variable time steps
    time_delta = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(batch_size, seq_length)
    )

    return X, y, time_delta

4. Add Performance Monitoring

.. code:: python

def train_with_monitoring(model, n_epochs=100):
    """Train model with performance monitoring."""
    import time

    optimizer = optim.Adam(learning_rate=0.01)
    metrics = {
        'loss': [],
        'time_per_epoch': [],
        'memory_usage': []
    }

    for epoch in range(n_epochs):
        start_time = time.time()

        # Training step
        X, y, time_delta = generate_variable_time_data(32, 20)
        loss, grads = loss_and_grad_fn(model, X, y, time_delta)
        optimizer.update(model, grads)

        # Record metrics
        metrics['loss'].append(float(loss))
        metrics['time_per_epoch'].append(time.time() - start_time)

    return metrics

5. Add Visualization Improvements

.. code:: python

def visualize_model_behavior(model, X, time_delta=None):
    """Visualize model predictions and internal states."""
    # Get predictions and states
    outputs, states = model.cfc(X, time_delta=time_delta)
    predictions = model.output_layer(outputs)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot predictions
    ax1.plot(X[0, :, 0], label='Input')
    ax1.plot(predictions[0, :, 0], label='Prediction')
    ax1.set_title('Sequence Prediction')
    ax1.legend()

    # Plot state evolution
    ax2.plot(states[0].T)
    ax2.set_title('State Evolution')
    plt.tight_layout()
    plt.show()

6. Add Hardware Utilization Section

.. code:: python

def profile_model_performance(model, batch_sizes=[32, 64, 128]):
    """Profile model performance across batch sizes."""
    results = []

    for batch_size in batch_sizes:
        X = mx.random.normal((batch_size, 20, 1))

        # Measure forward pass time
        start_time = time.time()
        _ = model(X)
        forward_time = time.time() - start_time

        results.append({
            'batch_size': batch_size,
            'forward_time': forward_time,
        })

    return results

7. Add Error Handling Examples

.. code:: python

def safe_model_call(model, X, time_delta=None):
    """Demonstrate proper error handling."""
    try:
        # Validate inputs
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {X.shape}")

        if time_delta is not None and time_delta.shape[:2] != X.shape[:2]:
            raise ValueError("Time delta shape mismatch")

        return model(X, time_delta=time_delta)

    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return None

9.2 Additional Example Notebooks Needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. MLX Advanced Features

- Custom wiring patterns
- Complex architectures
- Performance optimization
- Hardware utilization

2. MLX Migration Guide

- Framework comparison
- Code migration
- Performance tuning
- Best practices

3. MLX Debugging Guide

- Common issues
- Troubleshooting
- Performance analysis
- Memory management

4. MLX Integration Examples

- External libraries
- Custom modules
- Framework interop
- Deployment patterns

[Previous sections remain the same…]
