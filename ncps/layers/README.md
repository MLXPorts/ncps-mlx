# NCPS Layer System

A Keras 3.8-compatible implementation of continuous-time neural cells, providing a rich set of neuron types for different use cases.

## Installation

```bash
pip install ncps
```

## Available Cells

### Core Cells

#### CfCCell (Closed-form Continuous-time)
```python
from ncps.layers import CfCCell

cell = CfCCell(
    units=32,
    mode="default",  # "default", "pure", or "no_gate"
    activation="tanh",
    backbone_units=128,
    backbone_layers=1,
    backbone_dropout=0.1
)
```

#### LTCCell (Linear Time-invariant)
```python
from ncps.layers import LTCCell

cell = LTCCell(
    units=32,
    activation="tanh",
    backbone_units=None,
    backbone_layers=0,
    backbone_dropout=0.0
)
```

### Advanced Cells

#### CTGRUCell (Continuous-time GRU)
```python
from ncps.layers import CTGRUCell

cell = CTGRUCell(
    units=32,
    activation="tanh",
    recurrent_activation="sigmoid",
    backbone_units=None,
    backbone_layers=0,
    backbone_dropout=0.0
)
```

#### CTRNNCell (Continuous-time RNN)
```python
from ncps.layers import CTRNNCell

cell = CTRNNCell(
    units=32,
    activation="tanh",
    backbone_units=None,
    backbone_layers=0,
    backbone_dropout=0.0
)
```

#### ELTCCell (Enhanced LTC)
```python
from ncps.layers import ELTCCell, ODESolver

cell = ELTCCell(
    units=32,
    solver=ODESolver.RUNGE_KUTTA,  # or SEMI_IMPLICIT, EXPLICIT
    ode_unfolds=6,
    sparsity=0.5,
    activation="tanh",
    hidden_size=None,
    backbone_units=None,
    backbone_layers=0,
    backbone_dropout=0.0
)
```

## Usage Examples

### Basic Usage

```python
import keras
from ncps.layers import CfCCell, LTCCell

# Create model with CfC
model = keras.Sequential([
    keras.layers.RNN(CfCCell(32)),
    keras.layers.Dense(10)
])

# Create model with LTC
model = keras.Sequential([
    keras.layers.RNN(LTCCell(32)),
    keras.layers.Dense(10)
])
```

### Time-step Control

```python
# Create inputs with time steps
x = keras.ops.ones((batch_size, input_dim))
t = keras.ops.ones((batch_size, 1)) * 0.5  # Half time step

# Process with explicit time
output, state = cell([x, t], initial_state)
```

### Enhanced Features

#### ODE Solvers (ELTC)
```python
# Use different ODE solvers
cell = ELTCCell(32, solver=ODESolver.SEMI_IMPLICIT)
cell = ELTCCell(32, solver=ODESolver.EXPLICIT)
cell = ELTCCell(32, solver=ODESolver.RUNGE_KUTTA)
```

#### Backbone Networks
```python
# Add backbone network for enhanced processing
cell = CfCCell(
    32,
    backbone_units=128,
    backbone_layers=2,
    backbone_dropout=0.1
)
```

### Training

```python
# Create model
model = keras.Sequential([
    keras.layers.RNN(CfCCell(32)),
    keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer="adam",
    loss="mse"
)

# Train
model.fit(x_train, y_train, epochs=10)
```

## Testing

Run the test suite:

```bash
pytest ncps/tests/