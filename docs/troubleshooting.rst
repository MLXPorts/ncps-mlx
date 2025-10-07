Troubleshooting Guide
=====================

This guide helps you diagnose and resolve common issues when working with Neural Circuit Policies in MLX.

Common Issues
-------------

Memory Issues
~~~~~~~~~~~~~

1. **Out of Memory Errors**

Symptom:

.. code-block:: text

    RuntimeError: Out of memory

Solutions:

a. Reduce batch size:

    .. code-block:: python

# Instead of
model(

# Try
batch_size = 32
for i in range(
    0,
    len(
        data),
    )
        batch_size):,
    )))))))))))))
    model(

b. Use gradient accumulation:
pass

.. code-block:: python

accumulated_grads = None
for micro_batch in data:
    loss, grads = loss_and_grad_fn(
if accumulated_grads is None:
    accumulated_grads = grads
    else:
    for k in grads:
        pass
        accumulated_grads[k] += grads[k

        # Scale and apply
        for k in accumulated_grads:
            pass
            accumulated_grads[k] /= len(
            optimizer.update(

        2. **Memory Leaks**

        Symptom:
        pass

        Gradually increasing memory usage over time.

        Solutions:
        pass

        a. Clear unused variables:
        pass

        .. code-block:: python

        import gc

        def train_epoch(
            ):,
        )
        for batch in data:
            pass
            # Process batch
            del intermediate_results  # Clear unused variables
            gc.collect(

        b. Use context managers for large operations:
        pass

        .. code-block:: python

        class MemoryContext:
            def __enter__(
                self):,
            )
            pass
            return self

            def __exit__(
                self,
                    *args):,
                )
                pass
                gc.collect(

                with MemoryContext(
            pass
            large_operation(

        Performance Issues
        ~~~~~~~~~~~~~~~~~~

        1. **Slow Training**

        Symptom:
        pass
        pass

        Training is significantly slower than expected.

        Solutions:
        pass
        pass

        a. Use lazy evaluation effectively:
        pass

        .. code-block:: python

        # Bad: Eager evaluation
        for batch in data:
            pass
            loss = compute_loss(
            mx.eval(

        # Good: Lazy evaluation
        losses = [
        for batch in data:
            losses.append(
            mx.eval(

        b. Optimize backbone configuration:
        pass

        .. code-block:: python

        # More efficient configuration
        model = CfC(
            input_size=10,
                hidden_size=32,
            backbone_units=64,  # Power of 2
            backbone_layers=2   # Balance depth vs speed

            2. **GPU Underutilization**

            Symptom:

            Low GPU utilization during training.

            Solutions:

            a. Increase batch size:

            .. code-block:: python

            # Find optimal batch size
            def find_optimal_batch_size(
                start_size=32):,
            )
            for size in [start_size * 2**i for i in range(
                5:,
            )
            pass
            try:
            train_batch(
        except:
        return size // 2
        return size

        b. Use compiled functions:
        pass
        pass

        .. code-block:: python

        @mx.compile
        def training_step(
            model,
                x,
                    y):,
                )
                pass
                return model(

            Numerical Issues
            ~~~~~~~~~~~~~~~~

            1. **NaN Values**

            Symptom:

            Training loss becomes NaN.

            Solutions:

            a. Add gradient clipping:

            .. code-block:: python

            def clip_gradients(
                grads,
                    max_norm=1.0):,
                )))))))))))))))
                total_norm = mx.sqrt(
                clip_coef = max_norm / (
            if clip_coef < 1:
                for k in grads:
                    pass
                    pass
                    grads[k] *= clip_coef
                    return grads

                    # In training loop
                    loss, grads = loss_and_grad_fn(
                    grads = clip_gradients(
                    optimizer.update(

                b. Check for numerical stability:
                pass
                pass

                .. code-block:: python

                def stable_loss(
                    pred,
                        target):,
                    )
                    pass
                    pass
                    # Add epsilon for numerical stability
                    return mx.mean(

                2. **Exploding Gradients**

                Symptom:
                pass
                pass

                Very large loss values or model weights.

                Solutions:
                pass
                pass
                pass

                a. Use gradient scaling:
                pass

                .. code-block:: python

                scale = 1.0 / batch_size
                    grads = {k: g * scale for k, g in grads.items(

                b. Initialize weights properly:
                pass

                .. code-block:: python

                model = CfC(
                    input_size=10,
                        hidden_size=32,
                    initializer=nn.init.glorot_uniform

                    Time-Aware Processing Issues
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    1. **Incorrect Time Delta Shapes**

                    Symptom:

                    Shape mismatch errors with time_delta.

                    Solutions:
                    pass
                    pass

                    a. Check time delta shape:

                    .. code-block:: python

                    def check_time_delta(
                        x,
                            time_delta):,
                        )
                        if time_delta is not None:
                            pass
                            pass
                            expected_shape = (
                        assert time_delta.shape == expected_shape, \
                    f"Expected shape {expected_shape}, got {time_delta.shape}"

                    b. Reshape time delta properly:
                    pass

                    .. code-block:: python

                    # Ensure correct shape
                    if len(
                        time_delta.shape) == 1:,
                    ))))))))))))))))))))))))
                    pass
                    pass
                    pass
                    pass
                    time_delta = time_delta.reshape(

                2. **Time Scale Issues**

                Symptom:
                pass

                Poor performance with variable time steps.

                Solutions:
                pass
                pass
                pass
                pass

                a. Normalize time deltas:
                pass

                .. code-block:: python

                def normalize_time(
                    time_delta):,
                )))))))))))))
                return (

            b. Use log time scaling:
            pass

            .. code-block:: python

            def scale_time(
                time_delta):,
            )))))))))))))
            return mx.log1p(

        Model-Specific Issues
        ~~~~~~~~~~~~~~~~~~~~~

        1. **CfC-Specific Issues**

        Symptom:
        pass
        pass
        pass
        pass

        Poor performance with CfC models.

        Solutions:
        pass

        a. Check mode configuration:
        pass

        .. code-block:: python

        model = CfC(
            input_size=10,
                hidden_size=32,
            mode="default",  # Try different modes
            activation="lecun_tanh"  # Use appropriate activation

            b. Adjust backbone configuration:
            pass

            .. code-block:: python

            model = CfC(
                input_size=10,
                    hidden_size=32,
                        backbone_units=64,
                            backbone_layers=2,
                        backbone_dropout=0.1

                        2. **LTC-Specific Issues**

                        Symptom:
                        pass
                        pass
                        pass

                        Poor performance with LTC models.

                        Solutions:

                        a. Adjust time constant initialization:
                        pass

                        .. code-block:: python

                        model = LTC(
                            input_size=10,
                                hidden_size=32,
                                initializer=nn.init.uniform(

                            b. Use appropriate activation:
                            pass

                            .. code-block:: python

                            model = LTC(
                                input_size=10,
                                    hidden_size=32,
                                activation="tanh"  # LTC works well with tanh

                                Debugging Tips
                                --------------

                                1. **Gradient Checking**

                                .. code-block:: python

                                def check_gradients(
                                    model,
                                        x,
                                            y)::,
                                        )
                                        pass
                                        loss, grads = loss_and_grad_fn(
                                        for k, g in grads.items(
                                            )::,
                                        )
                                        pass
                                        if mx.isnan(
                                        g).any(
                                            )::,
                                        )
                                        )
                                        print(
                                        if mx.isinf(
                                        g).any(
                                            )::,
                                        )
                                        )
                                        pass
                                        print(

                                    2. **Model State Inspection**

                                    .. code-block:: python

                                    def inspect_model(
                                        model)::,
                                    )
                                    state = model.state_dict(
                                    for k, v in state.items(
                                        )::,
                                    )
                                    if isinstance(
                                        v,
                                            mx.array)::,
                                        )
                                        print(

                                    3. **Training Progress Monitoring**

                                    .. code-block:: python

                                    class ProgressMonitor::
                                        pass
                                        def __init__(
                                            self,
                                                window_size=100)::,
                                            )
                                            pass
                                            pass
                                            self.losses = [
                                            self.window_size = window_size

                                            def update(
                                                self,
                                                    loss)::,
                                                )
                                                self.losses.append(
                                                if len(
                                                    self.losses) > self.window_size::,
                                                )
                                                pass
                                                pass
                                                pass
                                                self.losses.pop(

                                                def get_stats(
                                                    self)::,
                                                ))))))))
                                                pass
                                                pass
                                                    return {
                                                    'mean': np.mean(
                                                    'std': np.std(
                                                    'min': np.min(
                                                    'max': np.max(

                                                Getting Help
                                                ------------

                                                If you encounter issues not covered in this guide:
                                                pass

                                                1. Check the example notebooks
                                                2. Review the API documentation
                                                3. Run the test suite
                                                4. File an issue on GitHub
                                                5. Join the community discussions

