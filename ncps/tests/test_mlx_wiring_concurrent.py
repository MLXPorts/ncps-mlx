"""Tests for concurrent operations with wiring patterns."""

import mlx.core as mx
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring, FullyConnected, Random, NCP, AutoNCP


def create_and_run_model(wiring_config):
    """Create and run a model with given wiring configuration."""
    if wiring_config['type'] == 'random':
        wiring = Random(
            units=wiring_config['units'],
            sparsity_level=wiring_config['sparsity']
        )
    elif wiring_config['type'] == 'ncp':
        wiring = AutoNCP(
            units=wiring_config['units'],
            output_size=wiring_config['output_size'],
            sparsity_level=wiring_config['sparsity']
        )
    else:
        raise ValueError(f"Unknown wiring type: {wiring_config['type']}")
    
    model = CfC(wiring=wiring)
    x = mx.random.normal((1, 10, 8))
    return model(x)


def train_model(model, steps=100):
    """Train model for given number of steps."""
    optimizer = mx.optimizers.Adam(learning_rate=0.001)
    x = mx.random.normal((1, 10, 8))
    y = mx.random.normal((1, 10, model.cell.wiring.output_dim))
    
    losses = []
    for _ in range(steps):
        def loss_fn(model, x, y):
            pred = model(x)
            return mx.mean((pred - y) ** 2)
        
        loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        losses.append(float(loss))
    
    return losses


def test_parallel_model_creation():
    """Test creating multiple models in parallel."""
    configs = [
        {'type': 'random', 'units': 100, 'sparsity': 0.5},
        {'type': 'random', 'units': 200, 'sparsity': 0.7},
        {'type': 'ncp', 'units': 300, 'output_size': 30, 'sparsity': 0.8},
        {'type': 'ncp', 'units': 400, 'output_size': 40, 'sparsity': 0.9}
    ]
    
    # Run in parallel threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(create_and_run_model, config) for config in configs]
        results = [future.result() for future in futures]
    
    # Check results
    assert len(results) == len(configs)
    for result in results:
        assert isinstance(result, mx.array)
        assert not mx.any(mx.isnan(result))


def test_parallel_training():
    """Test training multiple models in parallel."""
    # Create different models
    models = [
        CfC(Random(units=100, sparsity_level=0.5)),
        CfC(Random(units=200, sparsity_level=0.7)),
        LTC(AutoNCP(units=300, output_size=30, sparsity_level=0.8))
    ]
    
    # Train in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_model, model) for model in models]
        results = [future.result() for future in futures]
    
    # Check results
    assert len(results) == len(models)
    for losses in results:
        assert len(losses) > 0
        assert losses[-1] < losses[0]  # Should show some improvement


def test_shared_wiring():
    """Test using the same wiring pattern across multiple models."""
    # Create shared wiring
    wiring = Random(units=100, sparsity_level=0.5)
    
    # Create multiple models with the same wiring
    models = [CfC(wiring=wiring) for _ in range(3)]
    
    # Train in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_model, model) for model in models]
        results = [future.result() for future in futures]
    
    # Models should train independently despite shared wiring
    assert len(set([tuple(losses) for losses in results])) == len(models)


def test_concurrent_modifications():
    """Test concurrent modifications to wiring patterns."""
    wiring = Wiring(units=10)
    
    def add_synapses(start, end):
        """Add synapses to a range of neurons."""
        for i in range(start, end):
            wiring.add_synapse(i, (i + 1) % 10, 1)
        return wiring.synapse_count
    
    # Modify wiring concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(add_synapses, 0, 5)
        future2 = executor.submit(add_synapses, 5, 10)
        count1 = future1.result()
        count2 = future2.result()
    
    # Final count should match individual counts
    assert wiring.synapse_count == count1 + count2


def test_parallel_batch_processing():
    """Test processing multiple batches in parallel."""
    model = CfC(Random(units=100, sparsity_level=0.5))
    
    def process_batch(batch_size):
        """Process a single batch."""
        x = mx.random.normal((batch_size, 10, 8))
        return model(x)
    
    # Process different batch sizes in parallel
    batch_sizes = [1, 16, 32, 64]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_batch, size) for size in batch_sizes]
        results = [future.result() for future in futures]
    
    # Check results
    for result, size in zip(results, batch_sizes):
        assert result.shape[0] == size


def test_concurrent_serialization():
    """Test concurrent serialization operations."""
    wirings = [
        Random(units=100, sparsity_level=0.5),
        NCP(
            inter_neurons=50,
            command_neurons=30,
            motor_neurons=10,
            sensory_fanout=5,
            inter_fanout=5,
            recurrent_command_synapses=10,
            motor_fanin=5
        ),
        AutoNCP(units=200, output_size=20, sparsity_level=0.7)
    ]
    
    def serialize_deserialize(wiring):
        """Serialize and deserialize a wiring pattern."""
        config = wiring.get_config()
        restored = wiring.__class__.from_config(config)
        return restored
    
    # Serialize/deserialize concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(serialize_deserialize, w) for w in wirings]
        restored = [future.result() for future in futures]
    
    # Check restored wirings
    for original, restored in zip(wirings, restored):
        assert isinstance(restored, original.__class__)
        assert mx.array_equal(original.adjacency_matrix, restored.adjacency_matrix)


def test_stress_test():
    """Stress test with many concurrent operations."""
    def create_and_train():
        """Create and train a model."""
        wiring = Random(
            units=np.random.randint(50, 150),
            sparsity_level=np.random.uniform(0.3, 0.7)
        )
        model = CfC(wiring=wiring)
        losses = train_model(model, steps=50)
        return np.mean(losses)
    
    # Run many concurrent training sessions
    n_sessions = 10
    with ThreadPoolExecutor(max_workers=n_sessions) as executor:
        futures = [executor.submit(create_and_train) for _ in range(n_sessions)]
        results = [future.result() for future in futures]
    
    # All sessions should complete without error
    assert len(results) == n_sessions
    assert all(isinstance(r, float) for r in results)


def test_parallel_gradient_computation():
    """Test computing gradients in parallel."""
    model = CfC(Random(units=100, sparsity_level=0.5))
    optimizer = mx.optimizers.Adam(learning_rate=0.001)
    
    def compute_gradients(batch_idx):
        """Compute gradients for a batch."""
        x = mx.random.normal((16, 10, 8))
        y = mx.random.normal((16, 10, model.cell.wiring.output_dim))
        
        def loss_fn(model, x, y):
            pred = model(x)
            return mx.mean((pred - y) ** 2)
        
        loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
        return grads
    
    # Compute gradients in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_gradients, i) for i in range(4)]
        all_grads = [future.result() for future in futures]
    
    # All gradient computations should succeed
    assert len(all_grads) == 4
    for grads in all_grads:
        assert isinstance(grads, dict)
        assert len(grads) > 0
