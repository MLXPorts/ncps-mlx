Model Compression Guide
====================

This guide covers techniques for compressing Neural Circuit Policies using MLX for efficient deployment.

Weight Quantization
----------------

Basic Quantization
~~~~~~~~~~~~~~~

Quantize model weights to lower precision.

.. code-block:: python

    def quantize_weights(model, bits=8):
        """Quantize model weights to reduced precision."""
        state_dict = model.state_dict()
        quantized_state = {}
        
        for key, value in state_dict.items():
            if isinstance(value, mx.array):
                # Compute scale and zero point
                max_val = mx.max(value)
                min_val = mx.min(value)
                scale = (max_val - min_val) / (2**bits - 1)
                zero_point = -min_val / scale
                
                # Quantize
                quantized = mx.round(value / scale + zero_point)
                
                # Store quantization parameters
                quantized_state[key] = {
                    'data': quantized,
                    'scale': scale,
                    'zero_point': zero_point
                }
            else:
                quantized_state[key] = value
                
        return quantized_state

Dequantization
~~~~~~~~~~~~

Convert quantized weights back to floating point.

.. code-block:: python

    def dequantize_weights(quantized_state):
        """Dequantize weights back to floating point."""
        state_dict = {}
        
        for key, value in quantized_state.items():
            if isinstance(value, dict) and 'scale' in value:
                # Dequantize
                dequantized = (value['data'] - value['zero_point']) * value['scale']
                state_dict[key] = dequantized
            else:
                state_dict[key] = value
                
        return state_dict

Model Pruning
-----------

Magnitude-Based Pruning
~~~~~~~~~~~~~~~~~~~~

Prune weights based on magnitude.

.. code-block:: python

    def prune_weights(model, threshold=0.01):
        """Prune small weights below threshold."""
        state_dict = model.state_dict()
        pruned_state = {}
        
        for key, value in state_dict.items():
            if isinstance(value, mx.array):
                # Create mask for significant weights
                mask = mx.abs(value) > threshold
                
                # Apply mask
                pruned = value * mask
                
                pruned_state[key] = {
                    'data': pruned,
                    'mask': mask
                }
            else:
                pruned_state[key] = value
                
        return pruned_state

Structured Pruning
~~~~~~~~~~~~~~~

Prune entire neurons or channels.

.. code-block:: python

    def structured_pruning(model, prune_ratio=0.1):
        """Prune entire neurons based on importance."""
        state_dict = model.state_dict()
        pruned_state = {}
        
        # Compute neuron importance
        importance = compute_neuron_importance(model)
        
        # Determine threshold
        k = int(len(importance) * prune_ratio)
        threshold = sorted(importance)[k]
        
        # Prune neurons
        for key, value in state_dict.items():
            if isinstance(value, mx.array):
                if 'weight' in key:
                    mask = importance > threshold
                    pruned = value * mask.reshape(-1, 1)
                    pruned_state[key] = pruned
                else:
                    pruned_state[key] = value
                    
        return pruned_state

Knowledge Distillation
-------------------

Teacher-Student Training
~~~~~~~~~~~~~~~~~~~~

Train smaller model to mimic larger model.

.. code-block:: python

    class DistillationLoss(nn.Module):
        def __init__(self, temperature=2.0):
            super().__init__()
            self.temperature = temperature
            
        def __call__(self, student_logits, teacher_logits, labels):
            """Compute distillation loss."""
            # Soften probabilities
            soft_targets = nn.softmax(teacher_logits / self.temperature)
            soft_prob = nn.softmax(student_logits / self.temperature)
            
            # Distillation loss
            distillation_loss = mx.mean(
                -soft_targets * mx.log(soft_prob + 1e-6)
            )
            
            # Student loss
            student_loss = mx.mean((student_logits - labels) ** 2)
            
            return student_loss + distillation_loss

    def train_with_distillation(teacher, student, train_data, n_epochs=100):
        """Train student model with knowledge distillation."""
        optimizer = nn.Adam(learning_rate=0.001)
        distill_loss = DistillationLoss()
        
        for epoch in range(n_epochs):
            for batch in train_data:
                x, y, time_delta = batch
                
                # Get teacher predictions
                with mx.stop_gradient():
                    teacher_pred = teacher(x, time_delta=time_delta)
                
                def loss_fn(student, x, y, teacher_pred, dt):
                    student_pred = student(x, time_delta=dt)
                    return distill_loss(student_pred, teacher_pred, y)
                
                loss, grads = nn.value_and_grad(student, loss_fn)(
                    student, x, y, teacher_pred, time_delta
                )
                optimizer.update(student, grads)

Time-Aware Compression
------------------

Temporal Pruning
~~~~~~~~~~~~~

Prune based on temporal importance.

.. code-block:: python

    def temporal_pruning(model, time_series_data, threshold=0.1):
        """Prune weights based on temporal importance."""
        importance_scores = []
        
        # Compute temporal importance
        for batch in time_series_data:
            x, _, time_delta = batch
            scores = compute_temporal_importance(model, x, time_delta)
            importance_scores.append(scores)
            
        importance = mx.mean(mx.stack(importance_scores), axis=0)
        
        # Prune based on importance
        state_dict = model.state_dict()
        pruned_state = {}
        
        for key, value in state_dict.items():
            if isinstance(value, mx.array):
                mask = importance > threshold
                pruned_state[key] = value * mask
            else:
                pruned_state[key] = value
                
        return pruned_state

Backbone Compression
-----------------

Backbone Optimization
~~~~~~~~~~~~~~~~~

Compress backbone networks.

.. code-block:: python

    def compress_backbone(model, compression_ratio=0.5):
        """Compress backbone networks."""
        if not hasattr(model, 'backbone_layers'):
            return model
            
        compressed_layers = []
        for layer in model.backbone_layers:
            # Reduce units by compression ratio
            in_features = layer.weight.shape[1]
            out_features = int(layer.weight.shape[0] * compression_ratio)
            
            compressed = nn.Linear(in_features, out_features)
            compressed_layers.append(compressed)
            
        model.backbone_layers = compressed_layers
        return model

Deployment Optimization
-------------------

Model Serialization
~~~~~~~~~~~~~~~~

Efficient model serialization.

.. code-block:: python

    def serialize_compressed_model(model, path):
        """Serialize compressed model efficiently."""
        state = {
            'model_config': {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'compressed': True
            },
            'quantization': {
                'bits': 8,
                'state': quantize_weights(model)
            },
            'pruning': {
                'mask': get_pruning_mask(model)
            }
        }
        
        with open(path, 'w') as f:
            json.dump(state, f)

Inference Optimization
~~~~~~~~~~~~~~~~~~

Optimize for inference.

.. code-block:: python

    class OptimizedInference:
        def __init__(self, compressed_model):
            self.model = compressed_model
            self.compiled_forward = mx.compile(self.model.__call__)
            
        def __call__(self, x, time_delta=None):
            return self.compiled_forward(x, time_delta)

Best Practices
------------

1. **Compression Strategy**
   - Start with quantization
   - Apply pruning gradually
   - Use knowledge distillation for complex models

2. **Evaluation**
   - Monitor accuracy impact
   - Measure memory reduction
   - Test inference speed

3. **Deployment**
   - Optimize for target hardware
   - Consider latency requirements
   - Balance size and accuracy

Example Usage
-----------

Complete compression example:

.. code-block:: python

    # Original model
    model = CfC(input_size=10, hidden_size=32)
    
    # Quantization
    quantized_state = quantize_weights(model, bits=8)
    
    # Pruning
    pruned_state = prune_weights(model, threshold=0.01)
    
    # Knowledge distillation
    student = CfC(input_size=10, hidden_size=16)
    train_with_distillation(model, student, train_data)
    
    # Optimize for deployment
    optimized = OptimizedInference(student)
    
    # Save compressed model
    serialize_compressed_model(student, 'compressed_model.json')

Getting Help
----------

If you need compression assistance:

1. Check example notebooks
2. Review compression strategies
3. Consult MLX documentation
4. Join community discussions
