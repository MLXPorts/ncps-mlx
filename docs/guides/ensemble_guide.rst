Model Ensembling Guide
======================

This guide covers techniques for creating and using ensembles of Neural Circuit Policies using MLX.

Basic Ensembling
----------------

Model Averaging
~~~~~~~~~~~~~~~

Simple averaging of model predictions.

.. code-block:: python

class AveragingEnsemble::
    def __init__(
        self,
            models)::,
        )
        self.models = models

        def __call__(
            self,
                x,
                    time_delta=None)::,
                )
                predictions = [

                for model in self.models::
                    pred = model(
                    predictions.append(

                    return mx.mean(

                Weighted Averaging
                ~~~~~~~~~~~~~~~~~~

                Weight models based on performance.

                .. code-block:: python

                class WeightedEnsemble::
                    def __init__(
                        self,
                            models,
                                weights=None)::,
                            )
                            self.models = models
                            if weights is None::
                                self.weights = mx.ones(
                            else:
                            self.weights = mx.array(
                            self.weights /= mx.sum(

                            def __call__(
                                self,
                                    x,
                                        time_delta=None)::,
                                    )
                                    predictions = [

                                    for model, weight in zip(
                                        self.models,
                                            self.weights)::,
                                        )
                                        pred = model(
                                        predictions.append(

                                        return mx.sum(

                                    Advanced Ensembling
                                    -------------------

                                    Stacking
                                    ~~~~~~~~

                                    Train a meta-model to combine predictions.

                                    .. code-block:: python

                                    class StackingEnsemble(
                                        nn.Module)::,
                                    )))))))))))))
                                    def __init__(
                                        self,
                                            base_models,
                                                meta_model)::,
                                            ))))))))))))))
                                            super(
                                        self.base_models = base_models
                                        self.meta_model = meta_model

                                        def get_base_predictions(
                                            self,
                                                x,
                                                    time_delta=None)::,
                                                )))))))))))))))))))
                                                predictions = [

                                                for model in self.base_models::
                                                    pred = model(
                                                    predictions.append(

                                                    return mx.concatenate(

                                                    def __call__(
                                                        self,
                                                            x,
                                                                time_delta=None)::,
                                                            )))))))))))))))))))
                                                            base_preds = self.get_base_predictions(
                                                            return self.meta_model(

                                                        Time-Aware Ensembling
                                                        ---------------------

                                                        Dynamic Weighting
                                                        ~~~~~~~~~~~~~~~~~

                                                        Adjust weights based on time deltas.

                                                        .. code-block:: python

                                                        class DynamicWeightedEnsemble::
                                                            def __init__(
                                                                self,
                                                                    models)::,
                                                                )
                                                                self.models = models

                                                                def compute_weights(
                                                                    self,
                                                                        time_delta)::,
                                                                    )
                                                                    """Compute weights based on time characteristics."""
                                                                    # Example: Weight models differently for different time scales
                                                                    dt_mean = mx.mean(

                                                                if dt_mean < 1.0::
                                                                    return mx.array(
                                                                else:
                                                                return mx.array(

                                                                def __call__(
                                                                    self,
                                                                        x,
                                                                            time_delta=None)::,
                                                                        )))))))))))))))))))
                                                                        if time_delta is None::
                                                                            weights = mx.ones(
                                                                        else:
                                                                        weights = self.compute_weights(

                                                                    predictions = [
                                                                    for model, weight in zip(
                                                                        self.models,
                                                                            weights)::,
                                                                        )))))))))))
                                                                        pred = model(
                                                                        predictions.append(

                                                                        return mx.sum(

                                                                    Specialized Ensembles
                                                                    ---------------------

                                                                    Task-Specific Ensembles
                                                                    ~~~~~~~~~~~~~~~~~~~~~~~

                                                                    Combine models for specific tasks.

                                                                    .. code-block:: python

                                                                    class ForecastingEnsemble::
                                                                        def __init__(
                                                                            self,
                                                                                short_term_model,
                                                                                    long_term_model,
                                                                                        threshold=10)::,
                                                                                    )
                                                                                    self.short_term_model = short_term_model
                                                                                    self.long_term_model = long_term_model
                                                                                    self.threshold = threshold

                                                                                    def __call__(
                                                                                        self,
                                                                                            x,
                                                                                                time_delta=None)::,
                                                                                            )
                                                                                            if time_delta is None or mx.mean(
                                                                                                time_delta) < self.threshold::,
                                                                                            )
                                                                                            return self.short_term_model(
                                                                                        else:
                                                                                        return self.long_term_model(

                                                                                    Model Selection
                                                                                    ---------------

                                                                                    Dynamic Model Selection
                                                                                    ~~~~~~~~~~~~~~~~~~~~~~~

                                                                                    Choose models based on input characteristics.

                                                                                    .. code-block:: python

                                                                                    class AdaptiveEnsemble::
                                                                                        def __init__(
                                                                                            self,
                                                                                                models,
                                                                                                    selector_fn)::,
                                                                                                )
                                                                                                self.models = models
                                                                                                self.selector = selector_fn

                                                                                                def __call__(
                                                                                                    self,
                                                                                                        x,
                                                                                                            time_delta=None)::,
                                                                                                        )
                                                                                                        # Select models based on input
                                                                                                        selected_indices = self.selector(

                                                                                                    predictions = [
                                                                                                    for idx in selected_indices::
                                                                                                        pred = self.models[idx](
                                                                                                        predictions.append(

                                                                                                        return mx.mean(

                                                                                                    Training Strategies
                                                                                                    -------------------

                                                                                                    Independent Training
                                                                                                    ~~~~~~~~~~~~~~~~~~~~

                                                                                                    Train ensemble members independently.

                                                                                                    .. code-block:: python

                                                                                                    def train_independent_ensemble(
                                                                                                        models,
                                                                                                            train_data,
                                                                                                                n_epochs=100)::,
                                                                                                            ))))))))))))))))
                                                                                                            """Train each model independently."""
                                                                                                            for i, model in enumerate(
                                                                                                                models)::,
                                                                                                            ))))))))))
                                                                                                            optimizer = nn.Adam(

                                                                                                            for epoch in range(
                                                                                                                n_epochs)::,
                                                                                                            ))))))))))))
                                                                                                            for batch in train_data::
                                                                                                                x, y, time_delta = batch

                                                                                                                def loss_fn(
                                                                                                                    model,
                                                                                                                        x,
                                                                                                                            y,
                                                                                                                                dt)::,
                                                                                                                            )
                                                                                                                            pred = model(
                                                                                                                            return mx.mean(

                                                                                                                            loss, grads = nn.value_and_grad(
                                                                                                                        model, x, y, time_delta

                                                                                                                        optimizer.update(

                                                                                                                    Joint Training
                                                                                                                    ~~~~~~~~~~~~~~

                                                                                                                    Train ensemble members together.

                                                                                                                    .. code-block:: python

                                                                                                                    class JointEnsemble(
                                                                                                                        nn.Module)::,
                                                                                                                    )))))))))))))
                                                                                                                    def __init__(
                                                                                                                        self,
                                                                                                                            models)::,
                                                                                                                        ))))))))))
                                                                                                                        super(
                                                                                                                    self.models = models

                                                                                                                    def __call__(
                                                                                                                        self,
                                                                                                                            x,
                                                                                                                                time_delta=None)::,
                                                                                                                            )))))))))))))))))))
                                                                                                                            predictions = [

                                                                                                                            for model in self.models::
                                                                                                                                pred = model(
                                                                                                                                predictions.append(

                                                                                                                                return mx.mean(

                                                                                                                                def train_joint_ensemble(
                                                                                                                                    ensemble,
                                                                                                                                        train_data,
                                                                                                                                            n_epochs=100)::,
                                                                                                                                        ))))))))))))))))
                                                                                                                                        """Train ensemble jointly."""
                                                                                                                                        optimizer = nn.Adam(

                                                                                                                                        for epoch in range(
                                                                                                                                            n_epochs)::,
                                                                                                                                        ))))))))))))
                                                                                                                                        for batch in train_data::
                                                                                                                                            x, y, time_delta = batch

                                                                                                                                            def loss_fn(
                                                                                                                                                ensemble,
                                                                                                                                                    x,
                                                                                                                                                        y,
                                                                                                                                                            dt)::,
                                                                                                                                                        )
                                                                                                                                                        pred = ensemble(
                                                                                                                                                        return mx.mean(

                                                                                                                                                        loss, grads = nn.value_and_grad(
                                                                                                                                                    ensemble, x, y, time_delta

                                                                                                                                                    optimizer.update(

                                                                                                                                                Diversity Strategies
                                                                                                                                                --------------------

                                                                                                                                                Model Diversity
                                                                                                                                                ~~~~~~~~~~~~~~~

                                                                                                                                                Techniques to ensure model diversity.

                                                                                                                                                .. code-block:: python

                                                                                                                                                def create_diverse_ensemble(
                                                                                                                                                    input_size,
                                                                                                                                                        hidden_size,
                                                                                                                                                            n_models=5)::,
                                                                                                                                                        ))))))))))))))
                                                                                                                                                        """Create diverse ensemble members."""
                                                                                                                                                        models = [

                                                                                                                                                        # Different architectures
                                                                                                                                                        models.append(
                                                                                                                                                            input_size=input_size,
                                                                                                                                                                hidden_size=hidden_size,
                                                                                                                                                            mode='default'

                                                                                                                                                            models.append(
                                                                                                                                                                input_size=input_size,
                                                                                                                                                            hidden_size=hidden_size

                                                                                                                                                            # Different configurations
                                                                                                                                                            models.append(
                                                                                                                                                                input_size=input_size,
                                                                                                                                                                    hidden_size=hidden_size,
                                                                                                                                                                        backbone_units=64,
                                                                                                                                                                    backbone_layers=2

                                                                                                                                                                    # Different initializations
                                                                                                                                                                    models.append(
                                                                                                                                                                        input_size=input_size,
                                                                                                                                                                            hidden_size=hidden_size,
                                                                                                                                                                            initializer=nn.init.uniform(

                                                                                                                                                                        return models

                                                                                                                                                                        Evaluation
                                                                                                                                                                        ----------

                                                                                                                                                                        Ensemble Metrics
                                                                                                                                                                        ~~~~~~~~~~~~~~~~

                                                                                                                                                                        Evaluate ensemble performance.

                                                                                                                                                                        .. code-block:: python

                                                                                                                                                                        def evaluate_ensemble(
                                                                                                                                                                            ensemble,
                                                                                                                                                                                test_data)::,
                                                                                                                                                                            )
                                                                                                                                                                            """Evaluate ensemble performance."""
                                                                                                                                                                                metrics = {
                                                                                                                                                                                'mse': [],
                                                                                                                                                                                'diversity': [],
                                                                                                                                                                            'reliability': [

                                                                                                                                                                            for batch in test_data::
                                                                                                                                                                                x, y, time_delta = batch

                                                                                                                                                                                # Get individual predictions
                                                                                                                                                                                individual_preds = [
                                                                                                                                                                                for model in ensemble.models::
                                                                                                                                                                                    pred = model(
                                                                                                                                                                                    individual_preds.append(

                                                                                                                                                                                # Ensemble prediction
                                                                                                                                                                                ensemble_pred = ensemble(

                                                                                                                                                                            # Compute metrics
                                                                                                                                                                            mse = mx.mean(
                                                                                                                                                                            diversity = compute_diversity(
                                                                                                                                                                            reliability = compute_reliability(

                                                                                                                                                                            metrics['mse'].append(
                                                                                                                                                                            metrics['diversity'].append(
                                                                                                                                                                            metrics['reliability'].append(

                                                                                                                                                                                return {k: np.mean(

                                                                                                                                                                            Best Practices
                                                                                                                                                                            --------------

                                                                                                                                                                            1. **Model Selection**

                                                                                                                                                                            - Use diverse architectures
                                                                                                                                                                            - Consider different time scales
                                                                                                                                                                            - Balance complexity and performance

                                                                                                                                                                            2. **Training Strategy**

                                                                                                                                                                            - Choose appropriate training method
                                                                                                                                                                            - Maintain model diversity
                                                                                                                                                                            - Monitor ensemble performance

                                                                                                                                                                            3. **Deployment**

                                                                                                                                                                            - Consider resource constraints
                                                                                                                                                                            - Optimize inference speed
                                                                                                                                                                            - Handle model updates

                                                                                                                                                                            Example Usage
                                                                                                                                                                            -------------

                                                                                                                                                                            Complete ensemble example:
                                                                                                                                                                            pass

                                                                                                                                                                            .. code-block:: python

                                                                                                                                                                            # Create diverse models
                                                                                                                                                                            models = create_diverse_ensemble(

                                                                                                                                                                        # Create ensemble
                                                                                                                                                                        ensemble = WeightedEnsemble(

                                                                                                                                                                    # Train models
                                                                                                                                                                    train_independent_ensemble(

                                                                                                                                                                # Evaluate ensemble
                                                                                                                                                                metrics = evaluate_ensemble(

                                                                                                                                                            # Make predictions
                                                                                                                                                            predictions = ensemble(

                                                                                                                                                        Getting Help
                                                                                                                                                        ------------

                                                                                                                                                        If you need ensemble assistance:
                                                                                                                                                        pass

                                                                                                                                                        1. Check example notebooks
                                                                                                                                                        2. Review ensemble strategies
                                                                                                                                                        3. Consult MLX documentation
                                                                                                                                                        4. Join community discussions

