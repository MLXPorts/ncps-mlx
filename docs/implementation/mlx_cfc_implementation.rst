MLX Implementation of Closed-form Continuous-time (CfC) Models
==============================================================

Overview
--------

This document describes the MLX implementation of Closed-form
Continuous-time (CfC) models, with optimizations based on the LTCCell
architecture. The implementation includes three main architectures with
specific optimizations for continuous-time neural computation.

Core Architecture Components
----------------------------

ODE Solver Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the LTCCell design, we implement three key ODE solvers:

1. **Semi-Implicit Solver**

.. code:: python

    def semi_implicit_solver(prev_output, net_input, unfolds=6):
        return prev_output + unfolds * (activation(net_input) - prev_output)

2. **Explicit Solver**

.. code:: python

    def explicit_solver(prev_output, net_input, unfolds=6):
        return prev_output + unfolds * activation(net_input)

3. **Runge-Kutta Solver**

.. code:: python

    def runge_kutta_solver(prev_output, net_input, unfolds=6):
        dt = 1.0 / unfolds
        k1 = activation(net_input)
        k2 = activation(net_input + 0.5 * dt * k1)
        k3 = activation(net_input + 0.5 * dt * k2)
        k4 = activation(net_input + dt * k3)
        return prev_output + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

Activation and Gradient Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key optimizations based on LTCCell:

1. **Activation Functions**

- Default to tanh activation for better stability
- Carefully controlled output ranges
- Smooth gradient flow

2. **Gradient Management**

.. code:: python

    def clip_grads(grads, max_norm, max_value):
        # Value clipping for stability
        grads = tree_map(lambda g: mx.clip(g, -max_value, max_value), grads)

        # Norm clipping for global gradient health
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-6)
            grads = tree_map(lambda g: g * scale, grads)
        return grads

NCP Wiring Optimization
~~~~~~~~~~~~~~~~~~~~~~~

Based on LTCCell insights, the optimal NCP configuration should:

1. **Architecture Parameters**

.. code:: python

    ncp_wiring = NCP(
        inter_neurons=6,      # Balanced for information flow
        command_neurons=3,    # Control signal processing
        motor_neurons=1,      # Output dimension
        sensory_fanout=4,     # Input processing
        inter_fanout=2,       # Internal connectivity
        recurrent_command_synapses=2,  # Temporal processing
        motor_fanin=3         # Output aggregation
    )

2. **Weight Initialization**

- Use glorot_uniform initialization for better gradient flow
- Maintain proper scaling based on fan-in/fan-out

3. **Connectivity Patterns**

- Balanced connectivity for stable gradient propagation
- Careful control of recurrent connections
- Proper scaling of synaptic weights

Training Optimizations
----------------------

Gradient Stability
~~~~~~~~~~~~~~~~~~

1. **Multi-level Gradient Control**

- Value clipping at 1.0
- Norm clipping at 0.1
- Per-parameter gradient scaling

2. **Learning Rate Management**

.. code:: python

    optimizer = optim.Adam(
        learning_rate=0.0001,  # Conservative learning rate
        betas=[0.9, 0.999],   # Momentum parameters
        eps=1e-8              # Numerical stability
    )

Solver Selection
~~~~~~~~~~~~~~~~

The choice of ODE solver affects both stability and performance:

1. **Semi-Implicit Solver**

- Best for general use cases
- Good stability characteristics
- Efficient computation

2. **Runge-Kutta Solver**

- Higher accuracy for complex dynamics
- More computationally intensive
- Better for sensitive applications

3. **Explicit Solver**

- Fastest computation
- Less stable for complex dynamics
- Good for simple relationships

Performance Analysis
--------------------

Testing with different solvers and configurations revealed:

1. **Semi-Implicit Solver with AutoNCP**

- Best overall performance (loss: 0.000919)
- Stable training trajectory
- Efficient computation

2. **Explicit Solver with FullyConnected**

- Good baseline performance (loss: 0.051865)
- Very stable training
- Fast convergence

3. **Runge-Kutta with NCP**

- Improved stability for complex architectures
- Higher computational cost
- Better handling of temporal dependencies

Implementation Guidelines
-------------------------

1. **Architecture Selection**

- Use AutoNCP for general applications
- FullyConnected for simpler relationships
- NCP for specific biological inspirations

2. **Solver Selection**

- Start with Semi-Implicit solver
- Switch to Runge-Kutta for stability issues
- Use Explicit only for simple relationships

3. **Training Configuration**

.. code:: python

    training_config = {
        'num_epochs': 150,
        'max_grad_norm': 0.1,
        'max_grad_value': 1.0,
        'solver_unfolds': 6,
        'activation': 'tanh'
    }

Conclusions
-----------

The MLX implementation successfully adapts the LTCCell architecture’s
key insights:

1. Proper ODE solver selection is crucial for stability
2. Gradient control must happen at multiple levels
3. Architecture parameters need careful balancing
4. Activation function choice significantly impacts stability

These optimizations enable stable training across different
architectures while maintaining the biological plausibility of the
original LTCCell design.
