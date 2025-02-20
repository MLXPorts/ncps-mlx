Model Evaluation Guide
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
===================

This guide covers metrics and evaluation strategies for Neural Circuit Policies using MLX.

Time Series Metrics
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

Basic Metrics
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~~~
~~~~~~~~~~~

Standard metrics adapted for time series.

.. code-block:: python

    def compute_metrics(y_true, y_pred, time_delta=None):
        """Compute basic time series metrics."""
        # MSE with optional time weighting
        if time_delta is not None:
            weights = 1.0 / (time_delta + 1e-6)
            mse = mx.mean(weights * (y_true - y_pred) ** 2)
        else:
            mse = mx.mean((y_true - y_pred) ** 2)
            
        # MAE
        mae = mx.mean(mx.abs(y_true - y_pred))
        
        # RMSE
        rmse = mx.sqrt(mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }

Time-Aware Metrics
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

Metrics that consider temporal aspects.

.. code-block:: python

    class TimeAwareMetrics:
        def __init__(self):
            self.reset()
            
        def reset(self):
            self.errors = []
            self.time_deltas = []
            
        def update(self, y_true, y_pred, time_delta):
            error = mx.abs(y_true - y_pred)
            self.errors.append(error)
            self.time_deltas.append(time_delta)
            
        def compute(self):
            errors = mx.stack(self.errors)
            time_deltas = mx.stack(self.time_deltas)
            
            # Time-weighted error
            weights = 1.0 / (time_deltas + 1e-6)
            weighted_error = mx.mean(weights * errors)
            
            # Error vs time delta correlation
            correlation = compute_correlation(errors, time_deltas)
            
            return {
                'weighted_error': float(weighted_error),
                'error_time_correlation': float(correlation)
            }

Sequence Metrics
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

Multi-step Evaluation
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~

Evaluate multi-step predictions.

.. code-block:: python

    def evaluate_sequence(model, x, y, n_steps=5):
        """Evaluate multi-step predictions."""
        predictions = []
        current_x = x
        
        for _ in range(n_steps):
            pred = model(current_x)
            predictions.append(pred)
            current_x = update_sequence(current_x, pred)
            
        predictions = mx.stack(predictions, axis=1)
        
        # Compute metrics at each step
        step_metrics = []
        for i in range(n_steps):
            metrics = compute_metrics(y[:, i], predictions[:, i])
            step_metrics.append(metrics)
            
        return step_metrics

Sequence Alignment
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

Evaluate sequence alignment quality.

.. code-block:: python

    def compute_alignment(y_true, y_pred):
        """Compute sequence alignment metrics."""
        # Dynamic Time Warping distance
        dtw_distance = compute_dtw(y_true, y_pred)
        
        # Longest Common Subsequence
        lcs_ratio = compute_lcs(y_true, y_pred)
        
        return {
            'dtw_distance': float(dtw_distance),
            'lcs_ratio': float(lcs_ratio)
        }

Model Validation
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

Cross Validation
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

Time series cross validation strategies.

.. code-block:: python

    class TimeSeriesCV:
        def __init__(self, n_splits=5, test_size=0.2):
            self.n_splits = n_splits
            self.test_size = test_size
            
        def split(self, X, y=None):
            """Generate train/test splits preserving temporal order."""
            n_samples = len(X)
            indices = np.arange(n_samples)
            
            # Forward chaining split
            test_size = int(n_samples * self.test_size)
            for i in range(self.n_splits):
                train_end = n_samples - (i + 1) * test_size
                test_start = train_end
                test_end = test_start + test_size
                
                if train_end > 0:
                    yield (
                        indices[:train_end],
                        indices[test_start:test_end]
                    )

Uncertainty Estimation
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
----------------------
-------------------

Prediction Intervals
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

Compute prediction intervals.

.. code-block:: python

    def compute_prediction_intervals(model, x, n_samples=100, confidence=0.95):
        """Compute prediction intervals using Monte Carlo sampling."""
        predictions = []
        
        for _ in range(n_samples):
            # Forward pass with dropout enabled
            model.train()  # Enable dropout
            pred = model(x)
            predictions.append(pred)
            
        predictions = mx.stack(predictions)
        
        # Compute intervals
        lower = np.percentile(predictions, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 + confidence) / 2 * 100, axis=0)
        
        return lower, upper

Calibration Analysis
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

Evaluate prediction uncertainty calibration.

.. code-block:: python

    def evaluate_calibration(y_true, y_pred, uncertainties):
        """Evaluate uncertainty calibration."""
        # Compute calibration curve
        confidences = np.linspace(0, 1, 20)
        observed_frequencies = []
        
        for conf in confidences:
            intervals = compute_prediction_intervals(
                y_pred, uncertainties, confidence=conf
            )
            in_interval = (y_true >= intervals[0]) & (y_true <= intervals[1])
            observed_frequencies.append(np.mean(in_interval))
            
        # Compute calibration error
        calibration_error = np.mean(
            np.abs(np.array(observed_frequencies) - confidences)
        )
        
        return {
            'calibration_error': calibration_error,
            'confidences': confidences,
            'observed_frequencies': observed_frequencies
        }

Performance Analysis
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

Error Analysis
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

Analyze prediction errors.

.. code-block:: python

    class ErrorAnalyzer:
        def __init__(self):
            self.errors = []
            self.features = []
            
        def add_prediction(self, y_true, y_pred, features):
            error = mx.abs(y_true - y_pred)
            self.errors.append(error)
            self.features.append(features)
            
        def analyze(self):
            errors = mx.stack(self.errors)
            features = mx.stack(self.features)
            
            # Error distribution
            error_stats = {
                'mean': float(mx.mean(errors)),
                'std': float(mx.std(errors)),
                'median': float(mx.median(errors)),
                'max': float(mx.max(errors))
            }
            
            # Feature correlation
            correlations = compute_feature_correlations(errors, features)
            
            return {
                'error_stats': error_stats,
                'feature_correlations': correlations
            }

Visualization Tools
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

Time Series Plots
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

Visualization functions for time series evaluation.

.. code-block:: python

    def plot_predictions(y_true, y_pred, uncertainties=None):
        """Plot predictions with optional uncertainty bands."""
        plt.figure(figsize=(12, 6))
        
        # Plot true values
        plt.plot(y_true, 'b-', label='True')
        
        # Plot predictions
        plt.plot(y_pred, 'r--', label='Predicted')
        
        if uncertainties is not None:
            # Plot uncertainty bands
            plt.fill_between(
                range(len(y_pred)),
                y_pred - 2*uncertainties,
                y_pred + 2*uncertainties,
                color='r',
                alpha=0.2,
                label='95% Confidence'
            )
            
        plt.legend()
        plt.grid(True)
        plt.show()

Error Analysis Plots
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

Visualize error patterns.

.. code-block:: python

    def plot_error_analysis(errors, features, time_delta=None):
        """Plot error analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error distribution
        axes[0,0].hist(errors, bins=50)
        axes[0,0].set_title('Error Distribution')
        
        # Error vs features
        axes[0,1].scatter(features, errors)
        axes[0,1].set_title('Error vs Features')
        
        if time_delta is not None:
            # Error vs time delta
            axes[1,0].scatter(time_delta, errors)
            axes[1,0].set_title('Error vs Time Delta')
            
        # Autocorrelation
        axes[1,1].acorr(errors)
        axes[1,1].set_title('Error Autocorrelation')
        
        plt.tight_layout()
        plt.show()

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

1. **Evaluation Strategy**

   - Use appropriate time series splits
   - Consider temporal dependencies
   - Account for variable time steps

2. **Metric Selection**

   - Choose task-appropriate metrics
   - Include time-aware metrics
   - Consider uncertainty evaluation

3. **Validation Process**

   - Use proper cross-validation
   - Maintain temporal order
   - Account for data dependencies

4. **Error Analysis**

   - Analyze error patterns
   - Consider feature relationships
   - Evaluate temporal effects

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

Complete evaluation example:

.. code-block:: python

    def evaluate_model(model, data):
        # Initialize metrics
        metrics = TimeAwareMetrics()
        error_analyzer = ErrorAnalyzer()
        
        # Evaluate predictions
        for batch in data:
            x, y, time_delta = batch
            
            # Get predictions
            pred = model(x, time_delta=time_delta)
            
            # Update metrics
            metrics.update(y, pred, time_delta)
            error_analyzer.add_prediction(y, pred, x)
        
        # Compute final results
        results = {
            'metrics': metrics.compute(),
            'error_analysis': error_analyzer.analyze()
        }
        
        # Plot results
        plot_predictions(y, pred)
        plot_error_analysis(
            results['error_analysis']['errors'],
            x,
            time_delta
        )
        
        return results

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

If you need evaluation assistance:

1. Check example notebooks
2. Review metric definitions
3. Consult MLX documentation
4. Join community discussions
