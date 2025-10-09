"""Advanced profiling tools integrating with MLX's built-in profilers.

This module provides tools for detailed performance analysis and profiling of neural circuit models,
including compute, memory, and stream profiling capabilities.
"""

import time
from typing import Dict, List, Optional
import mlx.core as mx
import numpy as np


class MLXProfiler:
    """Advanced profiler integrating with MLX's built-in tools.
    
    The MLXProfiler provides comprehensive profiling capabilities for neural circuit models including:
    - Compute performance metrics (FLOPS, throughput)
    - Memory usage tracking
    - Stream operation profiling
    - Historical performance data
    
    Args:
        model: The neural network model to profile
        
    Attributes:
        history: Dictionary storing historical profiling data
        model: The model being profiled
    """
    
    def __init__(self, model: 'CfC'):  # type: ignore
        self.model = model
        self.wiring = model.wiring
        self.history: Dict[str, List[Dict[str, float]]] = {
            'compute': [],
            'memory': [],
            'stream': []
        }
    
    def profile_compute(
        self,
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile compute performance.
        
        Args:
            batch_size: Size of batches to process
            seq_length: Length of sequences
            input_size: Size of input features (inferred if None)
            num_runs: Number of profiling runs to average
            
        Returns:
            Dictionary containing compute statistics including:
                - time_mean: Average compute time per run
                - time_std: Standard deviation of compute times
                - tflops: Teraflops achieved
                - flops_mean: Average FLOPS per run
        """
        if input_size is None:
            input_size = self.wiring.input_dim or 8
        
        x = mx.random.normal((batch_size, seq_length, input_size))
        
        # Enable MLX compute profiling
        mx.enable_compute_profiling()
        
        # Warmup
        _ = self.model(x)
        mx.eval()
        
        # Profile runs
        times = []
        flops = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.model(x)
            mx.eval()
            times.append(time.time() - start)
            
            # Get compute stats
            stats = mx.compute_stats()
            flops.append(stats['flops'])
        
        # Disable profiling
        mx.disable_compute_profiling()
        
        stats = {
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
            'flops_mean': float(np.mean(flops)),
            'flops_std': float(np.std(flops)),
            'tflops': float(np.mean(flops)) / float(np.mean(times)) / 1e12
        }
        
        self.history['compute'].append(stats)
        return stats
    
    def profile_memory(
        self,
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Profile memory usage.
        
        Args:
            batch_size: Size of batches to process
            seq_length: Length of sequences
            input_size: Size of input features (inferred if None)
            
        Returns:
            Dictionary containing memory statistics including:
                - peak_usage: Peak memory usage in MB
                - current_usage: Current memory usage in MB
                - total_allocated: Total memory allocated in MB
                - total_freed: Total memory freed in MB
        """
        if input_size is None:
            input_size = self.wiring.input_dim or 8
        
        x = mx.random.normal((batch_size, seq_length, input_size))
        
        # Enable MLX memory profiling
        mx.enable_memory_profiling()
        
        # Forward pass
        _ = self.model(x)
        mx.eval()
        
        # Get memory stats
        stats = mx.memory_stats()
        
        # Disable profiling
        mx.disable_memory_profiling()
        
        memory_stats = {
            'peak_usage': float(stats['peak_usage']) / (1024 * 1024),  # MB
            'current_usage': float(stats['current_usage']) / (1024 * 1024),  # MB
            'total_allocated': float(stats['total_allocated']) / (1024 * 1024),  # MB
            'total_freed': float(stats['total_freed']) / (1024 * 1024)  # MB
        }
        
        self.history['memory'].append(memory_stats)
        return memory_stats
    
    def profile_stream(
        self,
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None
    ) -> Dict[str, float]:
        """Profile stream operations.
        
        Args:
            batch_size: Size of batches to process
            seq_length: Length of sequences
            input_size: Size of input features (inferred if None)
            
        Returns:
            Dictionary containing stream statistics including:
                - kernel_time: Total kernel execution time
                - memory_time: Total memory operation time
                - num_kernels: Number of kernels executed
                - num_transfers: Number of memory transfers
        """
        if input_size is None:
            input_size = self.wiring.input_dim or 8
        
        x = mx.random.normal((batch_size, seq_length, input_size))
        
        # Enable MLX stream profiling
        mx.enable_stream_profiling()
        
        # Forward and backward pass
        def loss_fn(model, x):
            pred = model(x)
            return mx.mean(pred ** 2)
        
        loss, grads = mx.value_and_grad(self.model, loss_fn)(self.model, x)
        mx.eval()
        
        # Get stream stats
        stats = mx.stream_stats()
        
        # Disable profiling
        mx.disable_stream_profiling()
        
        stream_stats = {
            'kernel_time': float(stats['kernel_time']),
            'memory_time': float(stats['memory_time']),
            'num_kernels': int(stats['num_kernels']),
            'num_transfers': int(stats['num_transfers'])
        }
        
        self.history['stream'].append(stream_stats)
        return stream_stats
    
    def profile_all(
        self,
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None,
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Run all profiling tools.
        
        Args:
            batch_size: Size of input batch
            seq_length: Length of input sequence
            input_size: Size of input features
            num_runs: Number of profiling runs
        
        Returns:
            Dictionary with all profiling statistics
        """
        compute_stats = self.profile_compute(
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size,
            num_runs=num_runs
        )
        
        memory_stats = self.profile_memory(
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size
        )
        
        stream_stats = self.profile_stream(
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size
        )
        
        return {
            'compute': compute_stats,
            'memory': memory_stats,
            'stream': stream_stats
        }
    
    def summary(self) -> str:
        """Generate profiling summary.
        
        Returns:
            Multi-line string containing formatted summary of all profiling data.
        """
        if not any(self.history.values()):
            return "No profiling data available."
        
        lines = ["MLX Profiling Summary", "===================="]
        
        # Compute summary
        if self.history['compute']:
            latest = self.history['compute'][-1]
            lines.extend([
                "",
                "Compute Performance",
                "------------------",
                f"Average time: {latest['time_mean']*1000:.2f} ms",
                f"TFLOPS: {latest['tflops']:.2f}",
                f"FLOPS: {latest['flops_mean']:,.0f}"
            ])
        
        # Memory summary
        if self.history['memory']:
            latest = self.history['memory'][-1]
            lines.extend([
                "",
                "Memory Usage",
                "------------",
                f"Peak usage: {latest['peak_usage']:.2f} MB",
                f"Current usage: {latest['current_usage']:.2f} MB",
                f"Total allocated: {latest['total_allocated']:.2f} MB",
                f"Total freed: {latest['total_freed']:.2f} MB"
            ])
        
        # Stream summary
        if self.history['stream']:
            latest = self.history['stream'][-1]
            lines.extend([
                "",
                "Stream Operations",
                "----------------",
                f"Kernel time: {latest['kernel_time']*1000:.2f} ms",
                f"Memory time: {latest['memory_time']*1000:.2f} ms",
                f"Number of kernels: {latest['num_kernels']}",
                f"Number of transfers: {latest['num_transfers']}"
            ])
        
        return "\n".join(lines)
    
    def plot_history(self, metrics: Optional[List[str]] = None):
        """Plot profiling history.
        
        Args:
            metrics: List of metrics to plot (defaults to all)
        """
        import matplotlib.pyplot as plt
        
        if metrics is None:
            metrics = ['compute', 'memory', 'stream']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if not self.history[metric]:
                continue
            
            # Get values for plotting
            if metric == 'compute':
                values = [x['tflops'] for x in self.history[metric]]
                ylabel = 'TFLOPS'
            elif metric == 'memory':
                values = [x['peak_usage'] for x in self.history[metric]]
                ylabel = 'Peak Memory (MB)'
            else:  # stream
                values = [x['kernel_time'] for x in self.history[metric]]
                ylabel = 'Kernel Time (s)'
            
            ax.plot(values)
            ax.set_title(f'{metric.title()} History')
            ax.set_xlabel('Profile Run')
            ax.set_ylabel(ylabel)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()


def quick_profile(
    model: 'CfC',  # type: ignore
    batch_size: int = 32,
    seq_length: int = 10,
    input_size: Optional[int] = None,
    num_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """Quick profiling helper for common profiling scenarios.
    
    Performs compute, memory and stream profiling in one call.
    
    Args:
        model: Model to profile
        batch_size: Batch size to use
        seq_length: Sequence length to use
        input_size: Input feature size (inferred if None)
        num_runs: Number of profiling runs
        
    Returns:
        Dictionary containing results from all profiling types
    """
    profiler = MLXProfiler(model)
    return profiler.profile_all(
        batch_size=batch_size,
        seq_length=seq_length,
        input_size=input_size,
        num_runs=num_runs
    )
