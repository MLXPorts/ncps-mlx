Model Ensembling Guide
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
======================
==================

This guide covers techniques for creating and using ensembles of Neural Circuit Policies using MLX.

Basic Ensembling
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
----------------
-------------

Model Averaging
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~

Simple averaging of model predictions.

.. code-block:: python

    class AveragingEnsemble:
        def __init__(self, models):
            self.models = models
            
        def __call__(self, x, time_delta=None):
            predictions = []
            
            for model in self.models:
                pred = model(x, time_delta=time_delta)
                predictions.append(pred)
                
            return mx.mean(mx.stack(predictions), axis=0)

Weighted Averaging
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~

Weight models based on performance.

.. code-block:: python

    class WeightedEnsemble:
        def __init__(self, models, weights=None):
            self.models = models
            if weights is None:
                self.weights = mx.ones(len(models)) / len(models)
            else:
                self.weights = mx.array(weights)
                self.weights /= mx.sum(self.weights)
                
        def __call__(self, x, time_delta=None):
            predictions = []
            
            for model, weight in zip(self.models, self.weights):
                pred = model(x, time_delta=time_delta)
                predictions.append(weight * pred)
                
            return mx.sum(mx.stack(predictions), axis=0)

Advanced Ensembling
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
----------------

Stacking
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~~
~~~~~~~

Train a meta-model to combine predictions.

.. code-block:: python

    class StackingEnsemble(nn.Module):
        def __init__(self, base_models, meta_model):
            super().__init__()
            self.base_models = base_models
            self.meta_model = meta_model
            
        def get_base_predictions(self, x, time_delta=None):
            predictions = []
            
            for model in self.base_models:
                pred = model(x, time_delta=time_delta)
                predictions.append(pred)
                
            return mx.concatenate(predictions, axis=-1)
            
        def __call__(self, x, time_delta=None):
            base_preds = self.get_base_predictions(x, time_delta)
            return self.meta_model(base_preds)

Time-Aware Ensembling
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
------------------

Dynamic Weighting
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~

Adjust weights based on time deltas.

.. code-block:: python

    class DynamicWeightedEnsemble:
        def __init__(self, models):
            self.models = models
            
        def compute_weights(self, time_delta):
            """Compute weights based on time characteristics."""
            # Example: Weight models differently for different time scales
            dt_mean = mx.mean(time_delta)
            
            if dt_mean < 1.0:
                return mx.array([0.6, 0.4])  # Favor short-term model
            else:
                return mx.array([0.4, 0.6])  # Favor long-term model
                
        def __call__(self, x, time_delta=None):
            if time_delta is None:
                weights = mx.ones(len(self.models)) / len(self.models)
            else:
                weights = self.compute_weights(time_delta)
                
            predictions = []
            for model, weight in zip(self.models, weights):
                pred = model(x, time_delta=time_delta)
                predictions.append(weight * pred)
                
            return mx.sum(mx.stack(predictions), axis=0)

Specialized Ensembles
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
---------------------
------------------

Task-Specific Ensembles
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~

Combine models for specific tasks.

.. code-block:: python

    class ForecastingEnsemble:
        def __init__(self, short_term_model, long_term_model, threshold=10):
            self.short_term_model = short_term_model
            self.long_term_model = long_term_model
            self.threshold = threshold
            
        def __call__(self, x, time_delta=None):
            if time_delta is None or mx.mean(time_delta) < self.threshold:
                return self.short_term_model(x, time_delta=time_delta)
            else:
                return self.long_term_model(x, time_delta=time_delta)

Model Selection
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
---------------
------------

Dynamic Model Selection
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~

Choose models based on input characteristics.

.. code-block:: python

    class AdaptiveEnsemble:
        def __init__(self, models, selector_fn):
            self.models = models
            self.selector = selector_fn
            
        def __call__(self, x, time_delta=None):
            # Select models based on input
            selected_indices = self.selector(x, time_delta)
            
            predictions = []
            for idx in selected_indices:
                pred = self.models[idx](x, time_delta=time_delta)
                predictions.append(pred)
                
            return mx.mean(mx.stack(predictions), axis=0)

Training Strategies
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
-------------------
----------------

Independent Training
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~

Train ensemble members independently.

.. code-block:: python

    def train_independent_ensemble(models, train_data, n_epochs=100):
        """Train each model independently."""
        for i, model in enumerate(models):
            optimizer = nn.Adam(learning_rate=0.001)
            
            for epoch in range(n_epochs):
                for batch in train_data:
                    x, y, time_delta = batch
                    
                    def loss_fn(model, x, y, dt):
                        pred = model(x, time_delta=dt)
                        return mx.mean((pred - y) ** 2)
                    
                    loss, grads = nn.value_and_grad(model, loss_fn)(
                        model, x, y, time_delta
                    )
                    optimizer.update(model, grads)

Joint Training
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~~~
~~~~~~~~~~~~

Train ensemble members together.

.. code-block:: python

    class JointEnsemble(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = models
            
        def __call__(self, x, time_delta=None):
            predictions = []
            
            for model in self.models:
                pred = model(x, time_delta=time_delta)
                predictions.append(pred)
                
            return mx.mean(mx.stack(predictions), axis=0)
            
    def train_joint_ensemble(ensemble, train_data, n_epochs=100):
        """Train ensemble jointly."""
        optimizer = nn.Adam(learning_rate=0.001)
        
        for epoch in range(n_epochs):
            for batch in train_data:
                x, y, time_delta = batch
                
                def loss_fn(ensemble, x, y, dt):
                    pred = ensemble(x, time_delta=dt)
                    return mx.mean((pred - y) ** 2)
                
                loss, grads = nn.value_and_grad(ensemble, loss_fn)(
                    ensemble, x, y, time_delta
                )
                optimizer.update(ensemble, grads)

Diversity Strategies
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
--------------------
-----------------

Model Diversity
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~

Techniques to ensure model diversity.

.. code-block:: python

    def create_diverse_ensemble(input_size, hidden_size, n_models=5):
        """Create diverse ensemble members."""
        models = []
        
        # Different architectures
        models.append(CfC(
            input_size=input_size,
            hidden_size=hidden_size,
            mode='default'
        ))
        
        models.append(LTC(
            input_size=input_size,
            hidden_size=hidden_size
        ))
        
        # Different configurations
        models.append(CfC(
            input_size=input_size,
            hidden_size=hidden_size,
            backbone_units=64,
            backbone_layers=2
        ))
        
        # Different initializations
        models.append(CfC(
            input_size=input_size,
            hidden_size=hidden_size,
            initializer=nn.init.uniform(-0.1, 0.1)
        ))
        
        return models

Evaluation
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
----------
--------

Ensemble Metrics
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~

Evaluate ensemble performance.

.. code-block:: python

    def evaluate_ensemble(ensemble, test_data):
        """Evaluate ensemble performance."""
        metrics = {
            'mse': [],
            'diversity': [],
            'reliability': []
        }
        
        for batch in test_data:
            x, y, time_delta = batch
            
            # Get individual predictions
            individual_preds = []
            for model in ensemble.models:
                pred = model(x, time_delta=time_delta)
                individual_preds.append(pred)
            
            # Ensemble prediction
            ensemble_pred = ensemble(x, time_delta=time_delta)
            
            # Compute metrics
            mse = mx.mean((ensemble_pred - y) ** 2)
            diversity = compute_diversity(individual_preds)
            reliability = compute_reliability(individual_preds, y)
            
            metrics['mse'].append(float(mse))
            metrics['diversity'].append(float(diversity))
            metrics['reliability'].append(float(reliability))
            
        return {k: np.mean(v) for k, v in metrics.items()}

Best Practices
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
--------------
------------

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
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-------------
-----------

Complete ensemble example:

.. code-block:: python

    # Create diverse models
    models = create_diverse_ensemble(input_size=10, hidden_size=32)
    
    # Create ensemble
    ensemble = WeightedEnsemble(models)
    
    # Train models
    train_independent_ensemble(models, train_data)
    
    # Evaluate ensemble
    metrics = evaluate_ensemble(ensemble, test_data)
    
    # Make predictions
    predictions = ensemble(x, time_delta=time_delta)

Getting Help
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
------------
----------

If you need ensemble assistance:

1. Check example notebooks
2. Review ensemble strategies
3. Consult MLX documentation
4. Join community discussions
