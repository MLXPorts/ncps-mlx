"""Tests for ELTC cell implementation."""

import pytest
import keras
import numpy as np
from ncps.layers.eltc import ELTCCell, ODESolver


def test_initialization():
    """Test ELTC initialization."""
    # Test default configuration
    cell = ELTCCell(32)
    assert cell.units == 32
    assert cell.solver == ODESolver.RUNGE_KUTTA
    assert cell.ode_unfolds == 6
    assert cell.sparsity == 0.5
    assert cell.hidden_size == 32
    
    # Test custom configuration
    cell = ELTCCell(
        64,
        solver="semi_implicit",
        ode_unfolds=8,
        sparsity=0.7,
        activation="relu",
        hidden_size=128,
        backbone_units=256,
        backbone_layers=2,
        backbone_dropout=0.1
    )
    assert cell.units == 64
    assert cell.solver == ODESolver.SEMI_IMPLICIT
    assert cell.ode_unfolds == 8
    assert cell.sparsity == 0.7
    assert cell.hidden_size == 128
    assert cell.backbone_units == 256
    assert cell.backbone_layers == 2
    assert cell.backbone_dropout == 0.1


def test_build():
    """Test build method."""
    # Test without backbone
    cell = ELTCCell(32)
    cell.build((None, 64))
    assert cell.built
    assert cell.input_dense is not None
    assert cell.recurrent_dense is not None
    
    # Test with backbone
    cell = ELTCCell(32, backbone_units=128, backbone_layers=1)
    cell.build((None, 64))
    assert cell.built
    assert cell.backbone_fn is not None


def test_solvers():
    """Test different ODE solvers."""
    batch_size = 8
    input_dim = 64
    units = 32
    
    # Create inputs
    x = keras.ops.ones((batch_size, input_dim))
    state = [keras.ops.zeros((batch_size, units))]
    
    # Test each solver
    solvers = [ODESolver.SEMI_IMPLICIT, ODESolver.EXPLICIT, ODESolver.RUNGE_KUTTA]
    outputs = []
    
    for solver in solvers:
        cell = ELTCCell(units, solver=solver)
        cell.build((None, input_dim))
        output, _ = cell(x, state)
        outputs.append(output)
        
        # Check shapes
        assert output.shape == (batch_size, units)
    
    # Different solvers should give different results
    for i in range(len(solvers)):
        for j in range(i + 1, len(solvers)):
            assert not keras.ops.allclose(outputs[i], outputs[j])


def test_sparsity():
    """Test sparsity constraints."""
    batch_size = 8
    input_dim = 64
    units = 32
    
    # Create inputs
    x = keras.ops.ones((batch_size, input_dim))
    state = [keras.ops.zeros((batch_size, units))]
    
    # Test different sparsity levels
    sparsities = [0.0, 0.5, 0.9]
    outputs = []
    
    for sparsity in sparsities:
        cell = ELTCCell(units, sparsity=sparsity)
        cell.build((None, input_dim))
        output, _ = cell(x, state)
        outputs.append(output)
    
    # Different sparsity levels should give different results
    for i in range(len(sparsities)):
        for j in range(i + 1, len(sparsities)):
            assert not keras.ops.allclose(outputs[i], outputs[j])


def test_time_steps():
    """Test behavior with different time steps."""
    batch_size = 8
    input_dim = 64
    units = 32
    
    # Create inputs
    x = keras.ops.ones((batch_size, input_dim))
    t = keras.ops.ones((batch_size, 1)) * 0.5  # Half timestep
    state = [keras.ops.zeros((batch_size, units))]
    
    # Create cell
    cell = ELTCCell(units)
    cell.build((None, input_dim))
    
    # Test with different time steps
    output1, state1 = cell([x, t], state)
    output2, state2 = cell([x, t * 2], state)  # Double the time step
    
    # Outputs should differ due to time scaling
    assert not keras.ops.allclose(output1, output2)


def test_training():
    """Test training functionality."""
    # Create model
    model = keras.Sequential([
        keras.layers.RNN(ELTCCell(32)),
        keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    
    # Create sample data
    batch_size = 8
    time_steps = 10
    features = 64
    x = np.random.normal(size=(batch_size, time_steps, features))
    y = np.random.normal(size=(batch_size, 1))
    
    # Should train without errors
    history = model.fit(x, y, epochs=1)
    assert len(history.history["loss"]) == 1


def test_serialization():
    """Test serialization."""
    # Create and build cell
    cell = ELTCCell(
        32,
        solver="semi_implicit",
        ode_unfolds=8,
        sparsity=0.7,
        hidden_size=64
    )
    cell.build((None, 64))
    
    # Get config
    config = keras.saving.serialize_keras_object(cell)
    
    # Reconstruct
    new_cell = keras.saving.deserialize_keras_object(config)
    
    # Should have same configuration
    assert new_cell.units == cell.units
    assert new_cell.solver == cell.solver
    assert new_cell.ode_unfolds == cell.ode_unfolds
    assert new_cell.sparsity == cell.sparsity
    assert new_cell.hidden_size == cell.hidden_size


def test_stateful():
    """Test stateful operation."""
    # Create stateful RNN
    cell = ELTCCell(32)
    rnn = keras.layers.RNN(cell, stateful=True)
    
    # Create inputs
    batch_size = 8
    time_steps = 10
    features = 64
    x = keras.ops.ones((batch_size, time_steps, features))
    
    # First call
    out1 = rnn(x)
    
    # Second call should use final state from first call
    out2 = rnn(x)
    
    # Outputs should differ due to state
    assert not keras.ops.allclose(out1, out2)