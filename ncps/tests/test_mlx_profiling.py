"""Tests for profiling tools."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Wiring, Random, NCP, AutoNCP
from ncps.mlx.profiling import WiringProfiler, profile_wiring


def test_memory_measurement():
    """Test memory usage measurement."""
    # Test with different sizes
    sizes = [10, 100, 1000]
    memories = []
    
    for size in sizes:
        wiring = Random(units=size, sparsity_level=0.5)
        profiler = WiringProfiler(wiring)
        memory = profiler._measure_memory()
        memories.append(memory)
        
        # Memory should be proportional to size squared
        if len(memories) > 1:
            ratio = memory / memories[0]
            expected_ratio = (size / sizes[0]) ** 2
            assert 0.5 * expected_ratio <= ratio <= 1.5 * expected_ratio


def test_sparsity_measurement():
    """Test sparsity measurement."""
    # Test with different sparsity levels
    levels = [0.1, 0.5, 0.9]
    
    for level in levels:
        wiring = Random(units=100, sparsity_level=level)
        profiler = WiringProfiler(wiring)
        measured = profiler._measure_sparsity()
        
        # Should be close to target sparsity
        assert abs(measured - level) < 0.1


def test_connectivity_analysis():
    """Test connectivity analysis."""
    # Create a wiring with known connectivity
    wiring = Wiring(units=5)
    # Create a chain: 0 -> 1 -> 2 -> 3 -> 4
    for i in range(4):
        wiring.add_synapse(i, i+1, 1)
    
    profiler = WiringProfiler(wiring)
    stats = profiler.analyze_connectivity()
    
    # Check basic stats
    assert stats['avg_in_degree'] == 0.8  # 4 connections / 5 neurons
    assert stats['avg_out_degree'] == 0.8
    assert stats['max_in_degree'] == 1
    assert stats['max_out_degree'] == 1
    assert stats['num_components'] == 5  # Each node is its own SCC
    assert stats['avg_path_length'] > 0


def test_performance_profiling():
    """Test performance profiling."""
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = WiringProfiler(wiring)
    
    # Test forward profiling
    fwd_stats = profiler.profile_forward(
        model,
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    
    assert 'mean' in fwd_stats
    assert 'std' in fwd_stats
    assert 'min' in fwd_stats
    assert 'max' in fwd_stats
    assert fwd_stats['min'] <= fwd_stats['mean'] <= fwd_stats['max']
    
    # Test backward profiling
    bwd_stats = profiler.profile_backward(
        model,
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    
    assert 'mean' in bwd_stats
    assert 'std' in bwd_stats
    assert 'min' in bwd_stats
    assert 'max' in bwd_stats
    assert bwd_stats['min'] <= bwd_stats['mean'] <= bwd_stats['max']


def test_history_tracking():
    """Test profiling history tracking."""
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = WiringProfiler(wiring)
    
    # Initial stats should be recorded
    assert len(profiler.history['memory']) == 1
    assert len(profiler.history['sparsity']) == 1
    
    # Profile multiple times
    for _ in range(3):
        profiler.profile_forward(model, num_runs=10)
        profiler.profile_backward(model, num_runs=10)
    
    # Check history
    assert len(profiler.history['forward_time']) == 3
    assert len(profiler.history['backward_time']) == 3


def test_component_analysis():
    """Test strongly connected component analysis."""
    wiring = Wiring(units=6)
    
    # Create two components: (0,1,2) cycle and (3,4,5) cycle
    wiring.add_synapse(0, 1, 1)
    wiring.add_synapse(1, 2, 1)
    wiring.add_synapse(2, 0, 1)
    
    wiring.add_synapse(3, 4, 1)
    wiring.add_synapse(4, 5, 1)
    wiring.add_synapse(5, 3, 1)
    
    profiler = WiringProfiler(wiring)
    components = profiler._find_components()
    
    assert len(components) == 2
    assert all(len(c) == 3 for c in components)


def test_path_length_calculation():
    """Test average path length calculation."""
    wiring = Wiring(units=4)
    
    # Create a chain: 0 -> 1 -> 2 -> 3
    wiring.add_synapse(0, 1, 1)
    wiring.add_synapse(1, 2, 1)
    wiring.add_synapse(2, 3, 1)
    
    profiler = WiringProfiler(wiring)
    avg_length = profiler._calculate_average_path_length()
    
    # Average of all finite paths:
    # (0->1: 1) + (0->2: 2) + (0->3: 3) +
    # (1->2: 1) + (1->3: 2) +
    # (2->3: 1) = 10 total / 6 paths = 1.666...
    assert abs(avg_length - 1.666) < 0.1


def test_quick_profile():
    """Test quick profiling function."""
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    
    # Profile without model
    perf_stats, conn_stats = profile_wiring(wiring)
    assert len(perf_stats) == 0  # No performance stats without model
    assert 'sparsity' in conn_stats
    
    # Profile with model
    perf_stats, conn_stats = profile_wiring(
        wiring,
        model=model,
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    assert 'forward_time' in perf_stats
    assert 'backward_time' in perf_stats
    assert 'memory_mb' in perf_stats


def test_summary_generation():
    """Test summary string generation."""
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = WiringProfiler(wiring)
    
    # Profile some operations
    profiler.profile_forward(model, num_runs=10)
    profiler.profile_backward(model, num_runs=10)
    
    # Generate summary
    summary = profiler.summary()
    
    # Check key information is present
    assert "Wiring Profile Summary" in summary
    assert "Units: 50" in summary
    assert "Memory usage" in summary
    assert "Sparsity" in summary
    assert "Connectivity Statistics" in summary
    assert "Performance History" in summary


def test_error_handling():
    """Test error handling in profiling tools."""
    wiring = Random(units=50, sparsity_level=0.5)
    profiler = WiringProfiler(wiring)
    
    # Test with invalid model
    with pytest.raises(AttributeError):
        profiler.profile_forward(None)
    
    # Test with invalid batch size
    model = CfC(wiring=wiring)
    with pytest.raises(ValueError):
        profiler.profile_forward(model, batch_size=0)
    
    # Test with invalid sequence length
    with pytest.raises(ValueError):
        profiler.profile_forward(model, seq_length=0)
