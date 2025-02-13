Hyperparameter Tuning Guide
=======================

This guide provides strategies for tuning hyperparameters in Neural Circuit Policies using MLX.

Core Parameters
-------------

Hidden Size
~~~~~~~~~~

The number of hidden units affects model capacity and computational cost.

.. code-block:: python

    # Rule of thumb: Start with 2-4x input size
    model = CfC(
        input_size=10,
        hidden_size=32,  # Try [16, 32, 64, 128]
    )

Guidelines:
- Small datasets: Use smaller hidden sizes (16-32)
- Complex tasks: Try larger sizes (64-256)
- Monitor validation loss for overfitting

Number of Layers
~~~~~~~~~~~~~

Deeper networks can learn more complex patterns but are harder to train.

.. code-block:: python

    model = CfC(
        input_size=10,
        hidden_size=32,
        num_layers=2,  # Try [1, 2, 3]
    )

Guidelines:
- Start with 1-2 layers
- Add layers if underfitting
- Use residual connections for deeper networks

Backbone Configuration
-------------------

Units and Layers
~~~~~~~~~~~~~

Backbone networks help with feature extraction.

.. code-block:: python

    model = CfC(
        input_size=10,
        hidden_size=32,
        backbone_units=64,   # Try [32, 64, 128]
        backbone_layers=2,   # Try [1, 2, 3]
        backbone_dropout=0.1  # Try [0.1, 0.2, 0.3]
    )

Guidelines:
- backbone_units: Usually 1.5-2x hidden_size
- backbone_layers: Start with 1-2
- Increase complexity if underfitting

Dropout Rate
~~~~~~~~~~

Controls regularization strength.

.. code-block:: python

    # Systematic search
    dropout_rates = [0.1, 0.2, 0.3, 0.4]
    best_rate = None
    best_val_loss = float('inf')
    
    for rate in dropout_rates:
        model = CfC(
            input_size=10,
            hidden_size=32,
            backbone_dropout=rate
        )
        # Train and validate
        if val_loss < best_val_loss:
            best_rate = rate
            best_val_loss = val_loss

Guidelines:
- Start with 0.1-0.2
- Increase if overfitting
- Decrease if underfitting

Time-Aware Parameters
------------------

Time Scale
~~~~~~~~~

Proper time scaling is crucial for time-aware processing.

.. code-block:: python

    def tune_time_scaling(time_delta):
        # Try different scaling approaches
        scalings = {
            'raw': time_delta,
            'log': mx.log1p(time_delta),
            'normalized': (time_delta - mx.mean(time_delta)) / (mx.std(time_delta) + 1e-6),
            'bounded': mx.tanh(time_delta)
        }
        return scalings

Guidelines:
- Use log scaling for widely varying time steps
- Normalize if time scales are consistent
- Consider domain knowledge

Model-Specific Parameters
----------------------

CfC Parameters
~~~~~~~~~~~~

Specific to Closed-form Continuous-time models.

.. code-block:: python

    # Mode selection
    modes = ['default', 'pure', 'no_gate']
    
    for mode in modes:
        model = CfC(
            input_size=10,
            hidden_size=32,
            mode=mode,
            activation='lecun_tanh'
        )
        # Train and evaluate

Guidelines:
- default: Best for most cases
- pure: Simpler dynamics
- no_gate: Faster but less expressive

LTC Parameters
~~~~~~~~~~~~

Specific to Liquid Time-Constant models.

.. code-block:: python

    model = LTC(
        input_size=10,
        hidden_size=32,
        activation='tanh',  # Important for LTC
        initializer=nn.init.uniform(-0.1, 0.1)
    )

Guidelines:
- Use tanh activation
- Initialize time constants carefully
- Consider stability constraints

Optimization Parameters
--------------------

Learning Rate
~~~~~~~~~~~

Critical for training stability and convergence.

.. code-block:: python

    def lr_search():
        lrs = [1e-4, 3e-4, 1e-3, 3e-3]
        results = {}
        
        for lr in lrs:
            optimizer = nn.Adam(learning_rate=lr)
            # Train for few epochs
            results[lr] = validate()
        
        return results

Guidelines:
- Start with 1e-3
- Use learning rate warmup
- Consider scheduling

Batch Size
~~~~~~~~~

Affects both training stability and speed.

.. code-block:: python

    def find_batch_size():
        sizes = [16, 32, 64, 128]
        gpu_util = []
        
        for size in sizes:
            try:
                # Train with size
                gpu_util.append(measure_utilization())
            except:
                break
        
        return sizes[np.argmax(gpu_util)]

Guidelines:
- Start with 32
- Increase if GPU underutilized
- Consider gradient accumulation

Systematic Tuning
---------------

Grid Search
~~~~~~~~~

Exhaustive search over parameter combinations.

.. code-block:: python

    def grid_search():
        params = {
            'hidden_size': [32, 64],
            'num_layers': [1, 2],
            'backbone_dropout': [0.1, 0.2],
            'learning_rate': [1e-3, 3e-3]
        }
        
        results = {}
        for hidden_size in params['hidden_size']:
            for num_layers in params['num_layers']:
                for dropout in params['backbone_dropout']:
                    for lr in params['learning_rate']:
                        model = CfC(
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            backbone_dropout=dropout
                        )
                        optimizer = nn.Adam(learning_rate=lr)
                        # Train and validate
                        results[f"{hidden_size}_{num_layers}_{dropout}_{lr}"] = validate()
        
        return results

Random Search
~~~~~~~~~~~

More efficient for high-dimensional spaces.

.. code-block:: python

    def random_search(n_trials=20):
        def sample_params():
            return {
                'hidden_size': np.random.choice([32, 64, 128]),
                'num_layers': np.random.choice([1, 2, 3]),
                'backbone_dropout': np.random.uniform(0.1, 0.3),
                'learning_rate': np.random.loguniform(1e-4, 1e-2)
            }
        
        results = {}
        for _ in range(n_trials):
            params = sample_params()
            model = CfC(**params)
            # Train and validate
            results[str(params)] = validate()
        
        return results

Best Practices
------------

1. **Start Simple**
   - Begin with default parameters
   - Add complexity gradually
   - Monitor validation metrics

2. **Systematic Approach**
   - Document all experiments
   - Use version control for configs
   - Keep track of random seeds

3. **Resource Management**
   - Start with small-scale experiments
   - Use parameter sharing when possible
   - Consider computational budget

4. **Validation Strategy**
   - Use proper cross-validation
   - Monitor multiple metrics
   - Consider domain-specific metrics

Example Configurations
-------------------

Time Series Forecasting
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    model = CfC(
        input_size=input_dim,
        hidden_size=64,
        num_layers=2,
        backbone_units=128,
        backbone_layers=2,
        backbone_dropout=0.2,
        mode='default',
        activation='lecun_tanh'
    )

Anomaly Detection
~~~~~~~~~~~~~~

.. code-block:: python

    model = LTC(
        input_size=input_dim,
        hidden_size=32,
        num_layers=1,
        backbone_units=64,
        backbone_layers=1,
        backbone_dropout=0.1,
        activation='tanh'
    )

Getting Help
----------

If you need help with hyperparameter tuning:

1. Check the example notebooks
2. Review model-specific guidelines
3. Consider automated tuning tools
4. Join community discussions
