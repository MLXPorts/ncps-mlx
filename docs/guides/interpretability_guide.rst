Model Interpretability Guide
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
============================
========================

This guide covers techniques for interpreting and understanding Neural Circuit Policies using MLX.

Time-Aware Feature Attribution
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
------------------------------
--------------------------

Temporal Importance
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~

Analyze feature importance over time.

.. code-block:: python

    def compute_temporal_importance(model, x, time_delta=None):
        """Compute feature importance across time steps."""
        importance_scores = []
        
        # Compute gradients with respect to input
        def loss_fn(x):
            return model(x, time_delta=time_delta)
            
        grads = mx.grad(loss_fn)(x)
        
        # Normalize gradients
        importance = mx.abs(grads * x)  # Input * gradient for feature attribution
        temporal_importance = mx.mean(importance, axis=0)  # Average over batch
        
        return temporal_importance

Time Delta Sensitivity
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~

Analyze model sensitivity to time steps.

.. code-block:: python

    def analyze_time_sensitivity(model, x, base_time_delta):
        """Analyze model sensitivity to time delta variations."""
        base_pred = model(x, time_delta=base_time_delta)
        
        # Test different time scales
        scales = [0.5, 1.0, 2.0, 5.0]
        sensitivities = []
        
        for scale in scales:
            scaled_delta = base_time_delta * scale
            pred = model(x, time_delta=scaled_delta)
            
            # Compute sensitivity
            sensitivity = mx.mean(mx.abs(pred - base_pred))
            sensitivities.append(float(sensitivity))
            
        return {
            'scales': scales,
            'sensitivities': sensitivities
        }

State Analysis
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
-----------

Hidden State Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~

Visualize hidden state dynamics.

.. code-block:: python

    def visualize_hidden_states(model, x, time_delta=None):
        """Visualize hidden state evolution."""
        states = []
        current_state = None
        
        # Collect states
        for t in range(x.shape[1]):
            output, new_state = model.cell(
                x[:, t],
                current_state if current_state is not None \
                    else mx.zeros((x.shape[0], model.hidden_size)),
                time=time_delta[:, t] if time_delta is not None else 1.0
            )
            states.append(new_state)
            current_state = new_state
            
        states = mx.stack(states, axis=1)
        
        # Dimensionality reduction for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        states_2d = pca.fit_transform(states.reshape(-1, states.shape[-1]))
        states_2d = states_2d.reshape(states.shape[0], states.shape[1], 2)
        
        return states_2d

State Transition Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~

Analyze state transition patterns.

.. code-block:: python

    class StateTransitionAnalyzer:
        def __init__(self, model):
            self.model = model
            self.transitions = []
            
        def add_sequence(self, x, time_delta=None):
            states = []
            current_state = None
            
            for t in range(x.shape[1]):
                output, new_state = self.model.cell(
                    x[:, t],
                    current_state if current_state is not None \
                        else mx.zeros((x.shape[0], self.model.hidden_size)),
                    time=time_delta[:, t] if time_delta is not None else 1.0
                )
                
                if current_state is not None:
                    self.transitions.append((current_state, new_state))
                    
                current_state = new_state
                
        def analyze_transitions(self):
            # Compute transition statistics
            magnitudes = []
            directions = []
            
            for prev, curr in self.transitions:
                # Transition magnitude
                magnitude = mx.sqrt(mx.sum((curr - prev) ** 2))
                magnitudes.append(magnitude)
                
                # Transition direction
                direction = (curr - prev) / (magnitude + 1e-6)
                directions.append(direction)
                
            return {
                'magnitude_mean': float(mx.mean(magnitudes)),
                'magnitude_std': float(mx.std(magnitudes)),
                'direction_consistency': float(
                    mx.mean(mx.abs(mx.mean(directions, axis=0)))
                )
            }

Backbone Analysis
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
-----------------
--------------

Feature Transformation
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~

Analyze backbone network transformations.

.. code-block:: python

    def analyze_backbone(model, x):
        """Analyze backbone network feature transformations."""
        # Get intermediate activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output)
            
        # Register hooks
        hooks = []
        for layer in model.backbone_layers:
            hooks.append(layer.register_forward_hook(hook_fn))
            
        # Forward pass
        _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Analyze activations
        activation_stats = []
        for layer_activation in activations:
            stats = {
                'mean': float(mx.mean(layer_activation)),
                'std': float(mx.std(layer_activation)),
                'sparsity': float(mx.mean(layer_activation == 0))
            }
            activation_stats.append(stats)
            
        return activation_stats

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

State Space Plots
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

Visualize model state space.

.. code-block:: python

    def plot_state_space(states_2d, time_delta=None):
        """Plot 2D state space visualization."""
        plt.figure(figsize=(10, 10))
        
        # Plot state trajectories
        for i in range(states_2d.shape[0]):
            trajectory = states_2d[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', alpha=0.5)
            
            # Add time information if available
            if time_delta is not None:
                time_points = time_delta[i].cumsum()
                plt.scatter(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    c=time_points,
                    cmap='viridis'
                )
                
        plt.colorbar(label='Time')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('State Space Trajectories')
        plt.grid(True)
        plt.show()

Feature Attribution Plots
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~

Visualize feature importance.

.. code-block:: python

    def plot_feature_importance(importance_scores, feature_names=None):
        """Plot feature importance visualization."""
        plt.figure(figsize=(12, 6))
        
        # Plot importance scores
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(importance_scores.shape[-1])]
            
        plt.imshow(
            importance_scores.T,
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label='Importance')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title('Feature Importance Over Time')
        plt.show()

Model Understanding
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

Interpretability Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~

1. **Local Interpretability**

   - Analyze specific predictions
   - Track state evolution
   - Examine time dependencies

2. **Global Interpretability**

   - Analyze overall patterns
   - Study feature interactions
   - Understand temporal dynamics

3. **Time-Aware Analysis**

   - Study time delta effects
   - Analyze temporal patterns
   - Examine state transitions

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

Complete interpretability example:

.. code-block:: python

    def interpret_model(model, x, time_delta=None):
        """Comprehensive model interpretation."""
        # Feature attribution
        importance = compute_temporal_importance(model, x, time_delta)
        plot_feature_importance(importance)
        
        # Time sensitivity
        sensitivity = analyze_time_sensitivity(model, x, time_delta)
        
        # State analysis
        states_2d = visualize_hidden_states(model, x, time_delta)
        plot_state_space(states_2d, time_delta)
        
        # Transition analysis
        analyzer = StateTransitionAnalyzer(model)
        analyzer.add_sequence(x, time_delta)
        transition_stats = analyzer.analyze_transitions()
        
        # Backbone analysis
        backbone_stats = analyze_backbone(model, x)
        
        return {
            'importance': importance,
            'sensitivity': sensitivity,
            'states': states_2d,
            'transitions': transition_stats,
            'backbone': backbone_stats
        }

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

1. **Comprehensive Analysis**

   - Combine multiple techniques
   - Consider temporal aspects
   - Validate interpretations

2. **Visualization**

   - Use clear visualizations
   - Include temporal information
   - Show uncertainty when applicable

3. **Validation**

   - Cross-validate findings
   - Compare with baselines
   - Consider edge cases

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

If you need interpretability assistance:

1. Check example notebooks
2. Review visualization guides
3. Consult MLX documentation
4. Join community discussions
