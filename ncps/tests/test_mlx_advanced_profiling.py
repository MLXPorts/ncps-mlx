"""Tests for advanced profiling tools."""

import mlx.core as mx
import numpy as np
import pytest
from ncps.mlx import CfC, LTC
from ncps.mlx.wirings import Random, NCP, AutoNCP
from ncps.mlx.advanced_profiling import MLXProfiler, quick_profile


def test_compute_profiling():
    """Test computation profiling."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Profile compute
    stats = profiler.profile_compute(
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    
    # Check stats
    assert 'time_mean' in stats
    assert 'time_std' in stats
    assert 'flops_mean' in stats
    assert 'flops_std' in stats
    assert 'tflops' in stats
    
    # Values should be reasonable
    assert 0 < stats['time_mean'] < 1  # Less than 1 second
    assert stats['time_std'] >= 0
    assert stats['flops_mean'] > 0
    assert stats['flops_std'] >= 0
    assert stats['tflops'] > 0


def test_memory_profiling():
    """Test memory profiling."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Profile memory
    stats = profiler.profile_memory(
        batch_size=16,
        seq_length=10
    )
    
    # Check stats
    assert 'peak_usage' in stats
    assert 'current_usage' in stats
    assert 'total_allocated' in stats
    assert 'total_freed' in stats
    
    # Values should be reasonable
    assert stats['peak_usage'] > 0
    assert stats['current_usage'] >= 0
    assert stats['total_allocated'] > 0
    assert stats['total_freed'] >= 0
    
    # Relationships should make sense
    assert stats['peak_usage'] >= stats['current_usage']
    assert stats['total_allocated'] >= stats['total_freed']


def test_stream_profiling():
    """Test stream profiling."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Profile stream
    stats = profiler.profile_stream(
        batch_size=16,
        seq_length=10
    )
    
    # Check stats
    assert 'kernel_time' in stats
    assert 'memory_time' in stats
    assert 'num_kernels' in stats
    assert 'num_transfers' in stats
    
    # Values should be reasonable
    assert stats['kernel_time'] > 0
    assert stats['memory_time'] >= 0
    assert stats['num_kernels'] > 0
    assert stats['num_transfers'] >= 0


def test_profile_all():
    """Test comprehensive profiling."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Profile everything
    stats = profiler.profile_all(
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    
    # Check all components
    assert 'compute' in stats
    assert 'memory' in stats
    assert 'stream' in stats
    
    # Each component should have appropriate stats
    assert 'time_mean' in stats['compute']
    assert 'peak_usage' in stats['memory']
    assert 'kernel_time' in stats['stream']


def test_history_tracking():
    """Test profiling history tracking."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Multiple profiling runs
    for _ in range(3):
        profiler.profile_all(
            batch_size=16,
            seq_length=10,
            num_runs=10
        )
    
    # Check history
    assert len(profiler.history['compute']) == 3
    assert len(profiler.history['memory']) == 3
    assert len(profiler.history['stream']) == 3


def test_quick_profile():
    """Test quick profiling function."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    
    # Quick profile
    stats = quick_profile(
        model,
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    
    # Check results
    assert 'compute' in stats
    assert 'memory' in stats
    assert 'stream' in stats


def test_different_models():
    """Test profiling with different model types."""
    models = [
        CfC(Random(units=50, sparsity_level=0.5)),
        LTC(Random(units=50, sparsity_level=0.5)),
        CfC(AutoNCP(units=50, output_size=10, sparsity_level=0.5))
    ]
    
    for model in models:
        profiler = MLXProfiler(model)
        stats = profiler.profile_all(
            batch_size=16,
            seq_length=10,
            num_runs=10
        )
        
        # All models should provide valid stats
        assert 'compute' in stats
        assert 'memory' in stats
        assert 'stream' in stats


def test_error_handling():
    """Test error handling in profiling tools."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Test with invalid batch size
    with pytest.raises(ValueError):
        profiler.profile_compute(batch_size=0)
    
    # Test with invalid sequence length
    with pytest.raises(ValueError):
        profiler.profile_compute(seq_length=0)
    
    # Test with invalid number of runs
    with pytest.raises(ValueError):
        profiler.profile_compute(num_runs=0)


def test_summary_generation():
    """Test summary string generation."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # No profiling data
    assert "No profiling data available" in profiler.summary()
    
    # After profiling
    profiler.profile_all(
        batch_size=16,
        seq_length=10,
        num_runs=10
    )
    summary = profiler.summary()
    
    # Check key sections
    assert "Compute Performance" in summary
    assert "Memory Usage" in summary
    assert "Stream Operations" in summary


def test_large_scale():
    """Test profiling with large models."""
    # Create large model
    wiring = Random(units=1000, sparsity_level=0.9)  # Sparse for memory efficiency
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Profile with large batch
    stats = profiler.profile_all(
        batch_size=128,
        seq_length=50,
        num_runs=5  # Fewer runs for large model
    )
    
    # Check that profiling completed successfully
    assert 'compute' in stats
    assert 'memory' in stats
    assert 'stream' in stats
    
    # Memory usage should be significant but not excessive
    assert 0 < stats['memory']['peak_usage'] < 10000  # Less than 10GB


def test_profiling_stability():
    """Test stability of profiling results."""
    # Create model
    wiring = Random(units=50, sparsity_level=0.5)
    model = CfC(wiring=wiring)
    profiler = MLXProfiler(model)
    
    # Multiple profiling runs
    results = []
    for _ in range(5):
        stats = profiler.profile_compute(
            batch_size=16,
            seq_length=10,
            num_runs=10
        )
        results.append(stats['time_mean'])
    
    # Results should be relatively stable
    mean = np.mean(results)
    std = np.std(results)
    cv = std / mean  # Coefficient of variation
    assert cv < 0.2  # Less than 20% variation
