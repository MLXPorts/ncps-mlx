MLX Scalar Operations and Array Handling
========================================

Overview
--------

When implementing neural networks in MLX, it’s crucial to use MLX’s
array operations instead of Python’s scalar operations to ensure proper
optimization and computation graph construction.

Key Principles
--------------

1. **Array Operations vs Scalar Operations**

.. code:: python

    # Incorrect - using Python scalar operations
    time = 1.0 / unfolds
    output = prev_output + time * activation(net_input)

    # Correct - using MLX array operations
    time = mx.array(1.0) / mx.array(unfolds)
    output = prev_output + time * mx.tanh(net_input)

2. **Broadcasting and Shape Handling**

.. code:: python

    # Incorrect - Python scalar broadcasting
    if isinstance(time, mx.array):
        time = time[:, None]  # Add dimension for broadcasting

    # Correct - MLX broadcasting
    time = mx.expand_dims(time, axis=1)

3. **Gradient Computation**

.. code:: python

    # Incorrect - mixing Python and MLX operations
    grad_norm = sum(mx.sum(g * g) for g in grads)

    # Correct - pure MLX operations
    grad_norm = mx.sum(mx.tree_map(lambda g: mx.sum(g * g), grads))

ODE Solver Implementation
-------------------------

The LTCCell implementation shows proper MLX scalar handling in ODE
solvers:

.. code:: python

def semi_implicit_solver(prev_output, net_input, unfolds):
    """Semi-implicit solver using proper MLX operations."""
    dt = mx.array(1.0) / mx.array(unfolds)
    return prev_output + dt * (mx.tanh(net_input) - prev_output)

def runge_kutta_solver(prev_output, net_input, unfolds):
    """Runge-Kutta solver with MLX operations."""
    dt = mx.array(1.0) / mx.array(unfolds)
    k1 = mx.tanh(net_input)
    k2 = mx.tanh(net_input + mx.multiply(dt * 0.5, k1))
    k3 = mx.tanh(net_input + mx.multiply(dt * 0.5, k2))
    k4 = mx.tanh(net_input + mx.multiply(dt, k3))
    return prev_output + mx.multiply(dt / 6.0, k1 + 2.0 * k2 + 2.0 * k3 + k4)

Gradient Clipping Implementation
--------------------------------

Proper MLX implementation of gradient clipping:

.. code:: python

def clip_grads(grads, max_norm, max_value):
    """Gradient clipping with MLX operations."""
    # Value clipping
    grads = mx.tree_map(
        lambda g: mx.clip(g, -max_value, max_value),
        grads
    )

    # Norm clipping
    grad_norm = mx.sqrt(
        mx.sum(
            mx.tree_map(
                lambda g: mx.sum(g * g),
                grads
            )
        )
    )

    scale = mx.where(
        grad_norm > max_norm,
        max_norm / (grad_norm + mx.array(1e-6)),
        mx.array(1.0)
    )

    return mx.tree_map(lambda g: g * scale, grads)

Activation Functions
--------------------

Proper MLX implementation of activation functions:

.. code:: python

def lecun_tanh(x):
    """LeCun's tanh activation with MLX operations."""
    return mx.multiply(mx.array(1.7159), mx.tanh(mx.multiply(mx.array(2.0/3.0), x)))

def gelu(x):
    """GELU activation with MLX operations."""
    return mx.multiply(
        x,
        mx.multiply(
            mx.array(0.5),
            mx.array(1.0) + mx.erf(x / mx.sqrt(mx.array(2.0)))
        )
    )

Parameter Updates
-----------------

Proper MLX implementation of parameter updates:

.. code:: python

def update_parameters(params, grads, lr):
    """Parameter updates with MLX operations."""
    return mx.tree_map(
        lambda p, g: p - mx.multiply(mx.array(lr), g),
        params,
        grads
    )

Best Practices
--------------

1. **Always Use MLX Operations**

- Use mx.array() for constants
- Use MLX’s mathematical operations
- Avoid Python arithmetic operators with scalars

2. **Proper Broadcasting**

- Use mx.expand_dims() instead of None indexing
- Use mx.broadcast_to() for explicit broadcasting
- Check shapes with .shape attribute

3. **Gradient Handling**

- Use mx.tree_map() for parameter updates
- Use MLX’s gradient clipping functions
- Keep computation graph pure with MLX operations

4. **Memory Efficiency**

- Use in-place operations where possible
- Avoid unnecessary array creation
- Use MLX’s lazy evaluation effectively

Common Pitfalls
---------------

1. **Mixing Python and MLX Operations**

.. code:: python

    # Wrong
    scale = 1.0 / (norm + 1e-6)

    # Correct
    scale = mx.array(1.0) / (norm + mx.array(1e-6))

2. **Incorrect Broadcasting**

.. code:: python

    # Wrong
    time_delta = time_delta[:, None]

    # Correct
    time_delta = mx.expand_dims(time_delta, axis=1)

3. **Non-MLX Math Operations**

.. code:: python

    # Wrong
    import math
    x = math.sqrt(2.0)

    # Correct
    x = mx.sqrt(mx.array(2.0))

Performance Considerations
--------------------------

1. **Lazy Evaluation**

- Use mx.eval() strategically
- Batch operations where possible
- Minimize graph materializations

2. **Memory Management**

- Reuse arrays when possible
- Clear unnecessary references
- Use MLX’s memory-efficient operations

3. **Computation Graph**

- Keep the graph simple
- Use MLX’s optimized operations
- Avoid unnecessary array creation

By following these guidelines, we ensure optimal performance and correct
behavior when using MLX for neural network implementations.
