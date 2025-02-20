# Keras-Style Core Neurons Design

## Overview

NCPS provides two core neuron types implemented as Keras cells:
1. CfC (Closed-form Continuous-time)
2. LTC (Linear Time-invariant Continuous-time)

Both follow Keras conventions while providing unique continuous-time processing capabilities.

## Base Infrastructure

```python
@keras.saving.register_keras_serializable(package="ncps")
class LiquidCell(keras.layers.Layer):
    """Base class for continuous-time cells."""
    
    def __init__(self, wiring, **kwargs):
        super().__init__(**kwargs)
        self.wiring = wiring
        self.units = wiring.units
        self.state_size = self.units
        self.output_size = wiring.output_dim or wiring.units
        self.input_size = wiring.input_dim
```

## Core Components

### 1. CfC Cell
```python
@keras.saving.register_keras_serializable(package="ncps")
class CfCCell(LiquidCell):
    """Closed-form Continuous-time cell.
    
    Modes:
    - default: Standard CfC with gating
    - pure: Pure closed-form solution
    - no_gate: Modified gating mechanism
    """
    
    def __init__(
        self,
        wiring,
        mode="default",
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs
    ):
        super().__init__(wiring, **kwargs)
        self.mode = mode
        self.activation = activation
```

### 2. LTC Cell
```python
@keras.saving.register_keras_serializable(package="ncps")
class LTCCell(LiquidCell):
    """Linear Time-invariant Continuous-time cell.
    
    Features:
    - Time-constant based updates
    - State evolution through ODE
    - Learnable time constants
    """
    
    def __init__(
        self,
        wiring,
        activation="tanh",
        backbone_units=None,
        backbone_layers=0,
        backbone_dropout=0.0,
        **kwargs
    ):
        super().__init__(wiring, **kwargs)
        self.activation = activation
```

## Common Features

### 1. Backbone Network
```python
def build_backbone(self):
    """Build backbone network layers."""
    if not self.backbone_layers:
        return None
        
    return keras.Sequential([
        keras.layers.Dense(units) for units in self.backbone_units
    ])
```

### 2. State Management
```python
def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get initial state for RNN processing."""
    return [
        keras.backend.zeros((batch_size, self.units))
    ]
```

### 3. Configuration Management
```python
def get_config(self):
    """Get configuration for serialization."""
    config = super().get_config()
    config.update({
        'wiring': self.wiring.get_config(),
        'activation': self.activation_name,
        'backbone_units': self.backbone_units,
        'backbone_layers': self.backbone_layers,
        'backbone_dropout': self.backbone_dropout,
    })
    return config
```

## Usage Examples

### 1. Basic Usage
```python
# Create wiring
wiring = ncps.wirings.NCP(
    input_size=10,
    output_size=1,
    units=32
)

# Create CfC model
model = keras.Sequential([
    keras.layers.RNN(
        CfCCell(
            wiring,
            mode='default',
            activation='tanh'
        ),
        return_sequences=True
    ),
    keras.layers.Dense(1)
])

# Create LTC model
model = keras.Sequential([
    keras.layers.RNN(
        LTCCell(
            wiring,
            activation='tanh'
        ),
        return_sequences=True
    ),
    keras.layers.Dense(1)
])
```

### 2. With Backbone
```python
cell = CfCCell(
    wiring,
    backbone_units=[64, 32],
    backbone_layers=2,
    backbone_dropout=0.1
)
```

## Implementation Details

### 1. CfC Modes

#### Default Mode
```python
def _gated_step(self, x):
    """Execute gated mode step."""
    ff1 = ops.matmul(x, self.ff1_kernel) + self.ff1_bias
    ff2 = ops.matmul(x, self.ff2_kernel) + self.ff2_bias
    
    t_a = self.time_a(x)
    t_b = self.time_b(x)
    t_interp = activations.sigmoid(-t_a + t_b)
    
    return ff1 * (1.0 - t_interp) + t_interp * ff2
```

#### Pure Mode
```python
def _pure_step(self, x):
    """Execute pure mode step."""
    ff1 = ops.matmul(x, self.ff1_kernel) + self.ff1_bias
    return -self.A * ops.exp(-(ops.abs(self.w_tau) + ops.abs(ff1))) * ff1 + self.A
```

### 2. LTC Update
```python
def call(self, inputs, states):
    """Process one step."""
    state = states[0]
    x = layers.concatenate([inputs, state])
    
    if self.backbone is not None:
        x = self.backbone(x)
    
    d = ops.matmul(x, self.kernel) + self.bias
    tau = ops.exp(self.tau_kernel(x))
    
    new_state = state + (-state + d) / tau
    new_state = self.activation(new_state)
    
    return new_state, [new_state]
```

## Testing Strategy

### 1. Unit Tests
```python
def test_cfc_modes():
    """Test CfC cell modes."""
    wiring = ncps.wirings.NCP(10, 1, 32)
    
    # Test default mode
    cell = CfCCell(wiring, mode='default')
    assert cell.mode == 'default'
    
    # Test pure mode
    cell = CfCCell(wiring, mode='pure')
    assert cell.mode == 'pure'
```

### 2. Integration Tests
```python
def test_with_keras():
    """Test integration with Keras."""
    model = keras.Sequential([
        keras.layers.RNN(CfCCell(wiring)),
        keras.layers.Dense(1)
    ])
    
    # Should work with standard Keras APIs
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train)
```

## Next Steps

1. Maintain existing functionality
2. Improve documentation
3. Add examples
4. Enhance testing
5. Consider optimizations