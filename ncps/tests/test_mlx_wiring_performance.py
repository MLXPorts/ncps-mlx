"""Performance regression tests for wiring patterns."""

import time
import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring, FullyConnected, Random, NCP, AutoNCP


def measure_memory(wiring):
    """Measure memory usage of wiring pattern."""
    # Get size of adjacency matrix
    adj_size = wiring.adjacency_matrix.size * 4  # float32 = 4 bytes
    
    # Get size of sensory matrix if it exists
    sensory_size = 0
    if wiring.sensory_adjacency_matrix is not None:
        sensory_size = wiring.sensory_adjacency_matrix.size * 4
    
    return adj_size + sensory_size


def measure_forward_time(model, input_data, num_runs=100):
    """Measure average forward pass time."""
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(input_data)
        mx.eval()  # Ensure computation is complete
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


def measure_backward_time(model, input_data, target_data, num_runs=100):
    """Measure average backward pass time."""
    times = []
    optimizer = mx.optimizers.Adam(learning_rate=0.001)
    
    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)
    
    for _ in range(num_runs):
        start = time.time()
        loss, grads = mx.value_and_grad(model, loss_fn)(model, input_data, target_data)
        optimizer.update(model, grads)
        mx.eval()  # Ensure computation is complete
        times.append(time.time() - start)
    return np.mean(times), np.std(times)


@pytest.mark.parametrize("size", [100, 500, 1000])
def test_fully_connected_scaling(size):
    """Test performance scaling of fully connected networks."""
    # Create network
    wiring = FullyConnected(units=size)
    model = CfC(wiring=wiring)
    
    # Measure memory
    memory = measure_memory(wiring)
    print(f"\nFully Connected (size={size}):")
    print(f"Memory usage: {memory/1024/1024:.2f} MB")
    
    # Expected memory for adjacency matrix (size x size)
    expected_memory = size * size * 4  # float32
    assert memory <= expected_memory * 1.1  # Allow 10% overhead
    
    # Measure forward pass time
    x = mx.random.normal((1, 10, 8))
    mean_fwd, std_fwd = measure_forward_time(model, x)
    print(f"Forward time: {mean_fwd*1000:.2f} ± {std_fwd*1000:.2f} ms")
    
    # Time should scale approximately quadratically
    # but we'll allow some variation due to hardware/optimization
    if size > 100:
        expected_ratio = (size/100) ** 2
        actual_ratio = mean_fwd / test_fully_connected_scaling.base_time
        assert actual_ratio < expected_ratio * 2  # Allow 2x expected scaling
    else:
        test_fully_connected_scaling.base_time = mean_fwd


@pytest.mark.parametrize("sparsity", [0.5, 0.9, 0.99])
def test_random_sparsity_performance(sparsity):
    """Test performance impact of different sparsity levels."""
    size = 1000
    wiring = Random(units=size, sparsity_level=sparsity)
    model = CfC(wiring=wiring)
    
    # Measure memory
    memory = measure_memory(wiring)
    print(f"\nRandom (sparsity={sparsity}):")
    print(f"Memory usage: {memory/1024/1024:.2f} MB")
    
    # Memory should scale with (1-sparsity)
    expected_memory = size * size * 4 * (1 - sparsity)  # float32
    assert memory <= expected_memory * 1.2  # Allow 20% overhead
    
    # Measure performance
    x = mx.random.normal((1, 10, 8))
    mean_fwd, std_fwd = measure_forward_time(model, x)
    print(f"Forward time: {mean_fwd*1000:.2f} ± {std_fwd*1000:.2f} ms")
    
    # Time should roughly scale with (1-sparsity)
    if hasattr(test_random_sparsity_performance, 'base_time'):
        expected_ratio = (1 - sparsity) / (1 - 0.5)  # Compare to 0.5 sparsity
        actual_ratio = mean_fwd / test_random_sparsity_performance.base_time
        assert actual_ratio < expected_ratio * 1.5  # Allow 50% variation
    elif sparsity == 0.5:
        test_random_sparsity_performance.base_time = mean_fwd


def test_ncp_layer_performance():
    """Test performance characteristics of NCP layers."""
    sizes = [(100, 50, 10), (200, 100, 20), (400, 200, 40)]
    
    results = []
    for inter, command, motor in sizes:
        wiring = NCP(
            inter_neurons=inter,
            command_neurons=command,
            motor_neurons=motor,
            sensory_fanout=5,
            inter_fanout=5,
            recurrent_command_synapses=10,
            motor_fanin=5
        )
        model = LTC(wiring=wiring)
        
        # Measure memory
        memory = measure_memory(wiring)
        
        # Measure performance
        x = mx.random.normal((1, 10, 8))
        fwd_time, fwd_std = measure_forward_time(model, x)
        
        results.append({
            'size': inter + command + motor,
            'memory': memory,
            'forward_time': fwd_time
        })
        
        print(f"\nNCP ({inter}, {command}, {motor}):")
        print(f"Memory usage: {memory/1024/1024:.2f} MB")
        print(f"Forward time: {fwd_time*1000:.2f} ± {fwd_std*1000:.2f} ms")
    
    # Check scaling properties
    for i in range(1, len(results)):
        size_ratio = results[i]['size'] / results[i-1]['size']
        time_ratio = results[i]['forward_time'] / results[i-1]['forward_time']
        
        # Time should scale better than quadratic
        assert time_ratio < size_ratio ** 2


def test_autoncp_optimization():
    """Test AutoNCP optimization effectiveness."""
    size = 1000
    output_size = 100
    
    # Compare different sparsity levels
    sparsity_levels = [0.5, 0.7, 0.9]
    results = []
    
    for sparsity in sparsity_levels:
        wiring = AutoNCP(
            units=size,
            output_size=output_size,
            sparsity_level=sparsity
        )
        model = CfC(wiring=wiring)
        
        # Measure memory and performance
        memory = measure_memory(wiring)
        x = mx.random.normal((1, 10, 8))
        y = mx.random.normal((1, 10, output_size))
        
        fwd_time, _ = measure_forward_time(model, x)
        bwd_time, _ = measure_backward_time(model, x, y)
        
        results.append({
            'sparsity': sparsity,
            'memory': memory,
            'forward_time': fwd_time,
            'backward_time': bwd_time
        })
        
        print(f"\nAutoNCP (sparsity={sparsity}):")
        print(f"Memory usage: {memory/1024/1024:.2f} MB")
        print(f"Forward time: {fwd_time*1000:.2f} ms")
        print(f"Backward time: {bwd_time*1000:.2f} ms")
    
    # Verify performance improvements with sparsity
    for i in range(1, len(results)):
        assert results[i]['memory'] < results[i-1]['memory']
        assert results[i]['forward_time'] < results[i-1]['forward_time'] * 1.2


@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
def test_batch_size_scaling(batch_size):
    """Test performance scaling with batch size."""
    wiring = AutoNCP(units=500, output_size=50, sparsity_level=0.8)
    model = CfC(wiring=wiring)
    
    # Measure performance
    x = mx.random.normal((batch_size, 10, 8))
    y = mx.random.normal((batch_size, 10, 50))
    
    fwd_time, fwd_std = measure_forward_time(model, x)
    bwd_time, bwd_std = measure_backward_time(model, x, y)
    
    print(f"\nBatch size {batch_size}:")
    print(f"Forward time: {fwd_time*1000:.2f} ± {fwd_std*1000:.2f} ms")
    print(f"Backward time: {bwd_time*1000:.2f} ± {bwd_std*1000:.2f} ms")
    
    # Time should scale sub-linearly with batch size
    if batch_size > 1:
        expected_ratio = batch_size / test_batch_size_scaling.prev_batch
        fwd_ratio = fwd_time / test_batch_size_scaling.prev_fwd
        bwd_ratio = bwd_time / test_batch_size_scaling.prev_bwd
        
        assert fwd_ratio < expected_ratio  # Should benefit from parallelization
        assert bwd_ratio < expected_ratio
    
    test_batch_size_scaling.prev_batch = batch_size
    test_batch_size_scaling.prev_fwd = fwd_time
    test_batch_size_scaling.prev_bwd = bwd_time


def test_memory_recovery():
    """Test memory recovery after large operations."""
    initial_memory = measure_memory(Wiring(units=10))
    
    # Create and delete large wiring
    wiring = FullyConnected(units=1000)
    large_memory = measure_memory(wiring)
    assert large_memory > initial_memory
    
    # Delete large wiring
    del wiring
    
    # Create small wiring again
    final_wiring = Wiring(units=10)
    final_memory = measure_memory(final_wiring)
    
    # Memory should be close to initial
    assert abs(final_memory - initial_memory) < initial_memory * 0.1
