===================================================
Training with MLX
===================================================

This guide shows how to train Neural Circuit Policy models using MLX's training patterns.

Basic Training Loop
===================

Here's a complete example of training a CfC model for sequence prediction:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from ncps import CfC
    from ncps.wirings import AutoNCP

    # Create synthetic training data
    def generate_data(batch_size, seq_len, input_size, output_size):
        """Generate random training data."""
        x = mx.random.normal((batch_size, seq_len, input_size))
        y = mx.random.normal((batch_size, seq_len, output_size))
        return x, y

    # Create model
    input_size = 20
    output_size = 4
    wiring = AutoNCP(28, output_size)
    model = CfC(input_size=input_size, units=wiring, return_sequences=True)

    # Define loss function
    def loss_fn(model, x, y):
        outputs, _ = model(x)
        return mx.mean((outputs - y) ** 2)

    # Create optimizer
    optimizer = optim.Adam(learning_rate=1e-3)

    # Get value and gradient function
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    num_epochs = 100
    batch_size = 32
    seq_len = 50

    for epoch in range(num_epochs):
        # Generate batch
        x_batch, y_batch = generate_data(batch_size, seq_len, input_size, output_size)
        
        # Compute loss and gradients
        loss, grads = value_and_grad_fn(model, x_batch, y_batch)
        
        # Update parameters
        optimizer.update(model, grads)
        
        # Evaluate to ensure updates are applied
        mx.eval(model.parameters(), optimizer.state)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}")

Sequence Classification
========================

Training a model to classify entire sequences:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from ncps import LTC
    from ncps.wirings import AutoNCP

    class SequenceClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
            super().__init__()
            wiring = AutoNCP(hidden_dim, hidden_dim)
            self.rnn = LTC(
                input_size=input_dim,
                units=wiring,
                return_sequences=False  # Only use last output
            )
            self.classifier = nn.Linear(self.rnn.output_size, num_classes)
        
        def __call__(self, x):
            features, _ = self.rnn(x)
            return self.classifier(features)

    # Create model
    model = SequenceClassifier(input_dim=20, hidden_dim=64, num_classes=10)

    # Loss function with cross-entropy
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction='mean')

    # Training setup
    optimizer = optim.Adam(learning_rate=1e-3)
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training loop
    for epoch in range(100):
        # Generate synthetic batch
        x = mx.random.normal((32, 50, 20))
        y = mx.random.randint(0, 10, (32,))
        
        # Train step
        loss, grads = value_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Time Series Forecasting
========================

Predicting future values in a time series:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from ncps import CfC
    from ncps.wirings import FullyConnected

    # Create model for forecasting
    input_size = 10
    hidden_size = 64
    forecast_steps = 5

    wiring = FullyConnected(units=hidden_size, output_dim=forecast_steps)
    wiring.build(input_size)
    model = CfC(input_size=input_size, units=wiring, return_sequences=False)

    # Loss function for forecasting
    def loss_fn(model, x, y_true):
        # x: historical data (batch, seq_len, features)
        # y_true: future values to predict (batch, forecast_steps)
        y_pred, _ = model(x)
        return mx.mean((y_pred - y_true) ** 2)

    # Training
    optimizer = optim.Adam(learning_rate=1e-3)
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(200):
        # Synthetic data: predict next 5 steps
        x = mx.random.normal((32, 100, input_size))
        y = mx.random.normal((32, forecast_steps))
        
        loss, grads = value_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Learning Rate Scheduling
=========================

Using a learning rate schedule for better training:

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from ncps import CfC

    # Create model
    model = CfC(input_size=20, units=50)

    # Define loss
    def loss_fn(model, x, y):
        outputs, _ = model(x)
        return mx.mean((outputs - y) ** 2)

    # Learning rate schedule
    def get_learning_rate(epoch, initial_lr=1e-3):
        """Exponential decay schedule."""
        decay_rate = 0.95
        return initial_lr * (decay_rate ** (epoch / 10))

    # Training with schedule
    value_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(100):
        # Update learning rate
        lr = get_learning_rate(epoch)
        optimizer = optim.Adam(learning_rate=lr)
        
        # Generate data
        x = mx.random.normal((32, 50, 20))
        y = mx.random.normal((32, 50, 50))
        
        # Train step
        loss, grads = value_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, LR: {lr:.6f}, Loss: {loss.item():.4f}")

Model Checkpointing
===================

Saving and loading model weights:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC
    from ncps.wirings import AutoNCP

    # Create and train model
    wiring = AutoNCP(28, 4)
    model = CfC(input_size=20, units=wiring)

    # ... training code ...

    # Save model weights
    weights = model.parameters()
    mx.save_safetensors("ncp_model.safetensors", dict(tree_flatten(weights)))

    # Load model weights
    loaded_weights = mx.load("ncp_model.safetensors")
    # Update model with loaded weights
    model.update(tree_unflatten(loaded_weights))

Evaluation
==========

Evaluating model performance:

.. code-block:: python

    import mlx.core as mx
    from ncps import CfC

    model = CfC(input_size=20, units=50)

    # Generate test data
    x_test = mx.random.normal((100, 50, 20))
    y_test = mx.random.normal((100, 50, 50))

    # Evaluate
    def evaluate(model, x, y):
        outputs, _ = model(x)
        mse = mx.mean((outputs - y) ** 2)
        return mse.item()

    test_loss = evaluate(model, x_test, y_test)
    print(f"Test MSE: {test_loss:.4f}")

Next Steps
==========

- Explore the `examples/ directory <https://github.com/SolaceHarmony/ncps-mlx/tree/master/examples>`_ for real-world applications
- See :doc:`../quickstart` for more details on model architectures
- Check out the :doc:`../api/index` for complete API reference
