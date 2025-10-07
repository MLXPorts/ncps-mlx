Migration Guide
===============

This guide helps you migrate your Neural Circuit Policies code to the MLX implementation, which is optimized for Apple Silicon.

Framework Differences
---------------------

MLX vs PyTorch
~~~~~~~~~~~~~~

1. **Model Creation**

PyTorch:

.. code-block:: python

from ncps.torch import CfC

model = CfC(
    input_size=10,
        hidden_size=32,
            num_layers=2,
        bidirectional=True

        MLX:
        pass

        .. code-block:: python

        from ncps.mlx import CfC, CfCCell
        from ncps.wirings import AutoNCP

        # Create wiring
        wiring = AutoNCP(

    # Create model
    model = CfC(
    cell=CfCCell(
        wiring=wiring,
            activation="tanh",
            backbone_units=[64, 64],
        backbone_layers=2
            ),
                bidirectional=True,
            merge_mode="concat"

            2. **Data Types**

            PyTorch:
            pass

            .. code-block:: python

            import torch

            x = torch.randn(
            x = x.to(

        MLX:
        pass

        .. code-block:: python

        import mlx.core as mx

        x = mx.random.normal(
    # MLX uses float32 by default

    3. **Training Loop**

    PyTorch:
    pass

    .. code-block:: python

    optimizer = torch.optim.Adam(

    for epoch in range(
        epochs)::,
    ))))))))))
    optimizer.zero_grad(
    output = model(
    loss = criterion(
    loss.backward(
    optimizer.step(

MLX:
pass

.. code-block:: python

optimizer = mx.optimizers.Adam(

def loss_fn(
    model,
        x,
            y)::,
        )))))
        pred = model(
        return mx.mean(

        loss_and_grad_fn = mx.value_and_grad(

        for epoch in range(
            epochs)::,
        ))))))))))
        loss, grads = loss_and_grad_fn(
        optimizer.update(

    MLX vs TensorFlow
    ~~~~~~~~~~~~~~~~~

    1. **Model Creation**

    TensorFlow:
    pass

    .. code-block:: python

    from ncps.tf import LTC

    model = tf.keras.Sequential(
    tf.keras.layers.InputLayer(
    LTC(

MLX:

.. code-block:: python

from ncps.mlx import LTC, LTCCell
from ncps.wirings import AutoNCP

wiring = AutoNCP(

model = LTC(
cell=LTCCell(
    wiring=wiring,
        activation="tanh",
    backbone_units=[64
        ),
    return_sequences=True

    2. **Data Processing**

    TensorFlow:

    .. code-block:: python

    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
    dataset = dataset.batch(

MLX:

.. code-block:: python

import mlx.core as mx

def create_batches(
    x,
        y,
            batch_size=32)::,
        )))))))))))))))))
        indices = mx.random.permutation(
        for i in range(
            0,
            len(
                x),
            )
                batch_size)::,
            ))))))))))))))
            batch_idx = indices[i:i + batch_size
            yield x[batch_idx], y[batch_idx

            3. **Training**

            TensorFlow:

            .. code-block:: python

            model.compile(
            model.fit(

        MLX:

        .. code-block:: python

        optimizer = mx.optimizers.Adam(

        @mx.compile(
        def train_step(
            model,
                training=True)::,
            )))))))))))))))))
            def loss_fn(
                model,
                    x,
                        y)::,
                    )))))
                    pred = model(
                    return mx.mean(
                    return mx.value_and_grad(

                Common Migration Tasks
                ----------------------

                1. **State Management**

                .. code-block:: python

                # MLX state handling
                class StatefulModel(
                    nn.Module)::,
                )))))))))))))
                def __init__(
                    self)::,
                ))))))))
                super(
                self.cfc = CfC(
                cell=CfCCell(
            return_state=True

            def __call__(
                self,
                    x,
                        state=None)::,
                    ))))))))))))))
                    pass
                    return self.cfc(

                2. **Custom Layers**

                .. code-block:: python

                # MLX custom layer
                class CustomLayer(
                    nn.Module)::,
                )))))))))))))
                def __init__(
                    self)::,
                ))))))))
                pass
                super(
                self.linear = nn.Linear(

                def __call__(
                    self,
                        x)::,
                    )))))
                    pass
                    return self.linear(

                3. **Data Loading**

                .. code-block:: python

                # MLX data loading
                class DataLoader::
                    pass
                    def __init__(
                        self,
                            x,
                                y,
                                    batch_size=32)::,
                                )
                                self.x = mx.array(
                                self.y = mx.array(
                            self.batch_size = batch_size

                            def __iter__(
                                self)::,
                            ))))))))
                            indices = mx.random.permutation(
                            for i in range(
                                0,
                                len(
                                    self.x),
                                )
                                    self.batch_size)::,
                                )))))))))))))))))))
                                pass
                                batch_idx = indices[i:i + self.batch_size
                                yield self.x[batch_idx], self.y[batch_idx

                                Best Practices
                                --------------

                                1. **Hardware Optimization**

                                - Use MLX's lazy evaluation
                                - Enable operator fusion
                                - Optimize batch sizes
                                - Monitor memory usage

                                2. **Code Structure**

                                - Separate model definition
                                - Use functional components
                                - Implement proper state management
                                - Handle device placement

                                3. **Performance**

                                - Profile code sections
                                - Use MLX compilation
                                - Optimize memory usage
                                - Monitor training metrics

                                Common Issues
                                -------------

                                1. **Memory Management**

                                - Clear unused variables
                                - Use appropriate batch sizes
                                - Monitor memory usage
                                - Implement checkpointing

                                2. **Performance**

                                - Enable MLX optimizations
                                - Profile bottlenecks
                                - Use efficient architectures
                                - Monitor hardware utilization

                                3. **Training**

                                - Implement proper logging
                                - Monitor gradients
                                - Track metrics
                                - Validate results

                                Getting Help
                                ------------

                                For migration assistance:

                                1. Check example notebooks
                                2. Review documentation
                                3. File GitHub issues
                                4. Join discussions

                                References
                                ----------

                                - `MLX Documentation <https://ml-explore.github.io/mlx/build/html/index.html>`_
                                - `PyTorch to MLX Guide <https://ml-explore.github.io/mlx/build/html/notebooks/pytorch_to_mlx.html>`_
                                - `TensorFlow to MLX Guide <https://ml-explore.github.io/mlx/build/html/notebooks/tensorflow_to_mlx.html>`_
                                - `Apple Silicon Optimization Guide <https://developer.apple.com/documentation/accelerate>`_

