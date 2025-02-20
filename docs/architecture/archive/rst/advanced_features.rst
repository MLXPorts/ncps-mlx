Advanced Features
=================

Time-Aware Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Variable time steps
    time_delta = mx.random.uniform(
        low=0.5,
        high=1.5,
        shape=(batch_size, seq_len)
    )
    
    # Process with time information
    outputs, states = model(x, time_delta=time_delta)

State Management
~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize states
    batch_size = 32
    initial_state = mx.zeros((batch_size, model.cell.units))
    
    # Process with explicit state
    outputs, final_state = model(x, initial_states=[initial_state])

Backbone Networks
~~~~~~~~~~~~~~~

.. code-block:: python

    # Create model with backbone layers
    model = CfC(
        cell=CfCCell(
            wiring=wiring,
            backbone_units=[64, 32],  # Two backbone layers
            backbone_layers=2,
            backbone_dropout=0.1
        ),
        return_sequences=True
    )

For more examples and advanced usage, see:
- examples/notebooks/mlx_cfc_example.ipynb
- examples/notebooks/mlx_ltc_rnn_example.ipynb
- examples/notebooks/mlx_advanced_profiling_guide.ipynb