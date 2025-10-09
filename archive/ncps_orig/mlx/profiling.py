"""Profiling tools for neural circuit policies."""

import time
from typing import Dict, List, Optional, Tuple, Union
import mlx.core as mx
import numpy as np
from ncps.mlx.wirings import Wiring


class WiringProfiler:
    """Profile wiring patterns for performance and analysis."""
    
    def __init__(self, wiring: Wiring):
        self.wiring = wiring
        self.history: Dict[str, List[float]] = {
            'memory': [],
            'forward_time': [],
            'backward_time': [],
            'sparsity': []
        }
        self._record_initial_stats()
    
    def _record_initial_stats(self):
        """Record initial wiring statistics."""
        self.history['memory'].append(self._measure_memory())
        self.history['sparsity'].append(self._measure_sparsity())
    
    def _measure_memory(self) -> float:
        """Measure memory usage in MB."""
        # Get size of adjacency matrix
        adj_size = self.wiring.adjacency_matrix.size * 4  # float32 = 4 bytes
        
        # Get size of sensory matrix if it exists
        sensory_size = 0
        if self.wiring.sensory_adjacency_matrix is not None:
            sensory_size = self.wiring.sensory_adjacency_matrix.size * 4
        
        return (adj_size + sensory_size) / (1024 * 1024)  # Convert to MB
    
    def _measure_sparsity(self) -> float:
        """Measure wiring sparsity."""
        total_possible = self.wiring.units * self.wiring.units
        actual_connections = self.wiring.synapse_count
        return 1.0 - (actual_connections / total_possible)
    
    def profile_forward(
        self,
        model: 'CfC',  # type: ignore
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile forward pass performance.
        
        Args:
            model: The model using this wiring
            batch_size: Batch size for test input
            seq_length: Sequence length for test input
            input_size: Input size (defaults to model's input_dim)
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with timing statistics
        """
        if input_size is None:
            input_size = self.wiring.input_dim or 8
        
        x = mx.random.normal((batch_size, seq_length, input_size))
        
        # Warmup
        _ = model(x)
        mx.eval()
        
        # Time forward passes
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = model(x)
            mx.eval()
            times.append(time.time() - start)
        
        stats = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times))
        }
        
        self.history['forward_time'].append(stats['mean'])
        return stats
    
    def profile_backward(
        self,
        model: 'CfC',  # type: ignore
        batch_size: int = 32,
        seq_length: int = 10,
        input_size: Optional[int] = None,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Profile backward pass performance.
        
        Args:
            model: The model using this wiring
            batch_size: Batch size for test input
            seq_length: Sequence length for test input
            input_size: Input size (defaults to model's input_dim)
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with timing statistics
        """
        if input_size is None:
            input_size = self.wiring.input_dim or 8
        
        x = mx.random.normal((batch_size, seq_length, input_size))
        y = mx.random.normal((batch_size, seq_length, self.wiring.output_dim))
        
        optimizer = mx.optimizers.Adam(learning_rate=0.001)
        
        def loss_fn(model, x, y):
            pred = model(x)
            return mx.mean((pred - y) ** 2)
        
        # Warmup
        loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
        optimizer.update(model, grads)
        mx.eval()
        
        # Time backward passes
        times = []
        for _ in range(num_runs):
            start = time.time()
            loss, grads = mx.value_and_grad(model, loss_fn)(model, x, y)
            optimizer.update(model, grads)
            mx.eval()
            times.append(time.time() - start)
        
        stats = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times))
        }
        
        self.history['backward_time'].append(stats['mean'])
        return stats
    
    def analyze_connectivity(self) -> Dict[str, Union[float, List[int]]]:
        """Analyze wiring connectivity patterns.
        
        Returns:
            Dictionary with connectivity statistics
        """
        # Get adjacency matrix
        adj = self.wiring.adjacency_matrix
        
        # Calculate in/out degrees
        in_degrees = mx.sum(mx.abs(adj), axis=0)
        out_degrees = mx.sum(mx.abs(adj), axis=1)
        
        # Find strongly connected components
        components = self._find_components()
        
        # Calculate path lengths
        avg_path_length = self._calculate_average_path_length()
        
        return {
            'sparsity': self._measure_sparsity(),
            'avg_in_degree': float(mx.mean(in_degrees)),
            'avg_out_degree': float(mx.mean(out_degrees)),
            'max_in_degree': int(mx.max(in_degrees)),
            'max_out_degree': int(mx.max(out_degrees)),
            'num_components': len(components),
            'component_sizes': [len(c) for c in components],
            'avg_path_length': avg_path_length
        }
    
    def _find_components(self) -> List[List[int]]:
        """Find strongly connected components using Kosaraju's algorithm."""
        adj = self.wiring.adjacency_matrix
        n = self.wiring.units
        visited = [False] * n
        stack = []
        components = []
        
        def dfs1(v):
            visited[v] = True
            for u in range(n):
                if adj[v, u] != 0 and not visited[u]:
                    dfs1(u)
            stack.append(v)
        
        def dfs2(v, component):
            visited[v] = True
            component.append(v)
            for u in range(n):
                if adj[u, v] != 0 and not visited[u]:
                    dfs2(u, component)
        
        # First DFS
        for v in range(n):
            if not visited[v]:
                dfs1(v)
        
        # Reset visited
        visited = [False] * n
        
        # Second DFS
        while stack:
            v = stack.pop()
            if not visited[v]:
                component = []
                dfs2(v, component)
                components.append(component)
        
        return components
    
    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length using Floyd-Warshall."""
        adj = self.wiring.adjacency_matrix
        n = self.wiring.units
        
        # Initialize distances
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        
        # Set direct connections
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    dist[i, j] = 1
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Calculate average of finite paths
        finite_paths = dist[~np.isinf(dist)]
        if len(finite_paths) > 0:
            return float(np.mean(finite_paths))
        return float('inf')
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get profiling history."""
        return self.history
    
    def plot_history(self, metrics: Optional[List[str]] = None):
        """Plot profiling history.
        
        Args:
            metrics: List of metrics to plot (defaults to all)
        """
        import matplotlib.pyplot as plt
        
        if metrics is None:
            metrics = list(self.history.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            values = self.history[metric]
            ax.plot(values)
            ax.set_title(f'{metric.replace("_", " ").title()} History')
            ax.set_xlabel('Update')
            ax.set_ylabel(metric)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> str:
        """Generate a summary of the wiring profile."""
        connectivity = self.analyze_connectivity()
        
        summary = [
            "Wiring Profile Summary",
            "=====================",
            f"Units: {self.wiring.units}",
            f"Output dimension: {self.wiring.output_dim}",
            f"Memory usage: {self._measure_memory():.2f} MB",
            f"Sparsity: {connectivity['sparsity']:.2%}",
            "",
            "Connectivity Statistics",
            "---------------------",
            f"Average in-degree: {connectivity['avg_in_degree']:.2f}",
            f"Average out-degree: {connectivity['avg_out_degree']:.2f}",
            f"Maximum in-degree: {connectivity['max_in_degree']}",
            f"Maximum out-degree: {connectivity['max_out_degree']}",
            f"Number of components: {connectivity['num_components']}",
            f"Average path length: {connectivity['avg_path_length']:.2f}",
            "",
            "Performance History",
            "------------------",
            f"Latest forward time: {self.history['forward_time'][-1]*1000:.2f}ms" if self.history['forward_time'] else "No forward passes recorded",
            f"Latest backward time: {self.history['backward_time'][-1]*1000:.2f}ms" if self.history['backward_time'] else "No backward passes recorded"
        ]
        
        return "\n".join(summary)


def profile_wiring(
    wiring: Wiring,
    model: Optional['CfC'] = None,  # type: ignore
    batch_size: int = 32,
    seq_length: int = 10,
    input_size: Optional[int] = None,
    num_runs: int = 100
) -> Tuple[Dict[str, float], Dict[str, Union[float, List[int]]]]:
    """Quick profile of a wiring pattern.
    
    Args:
        wiring: Wiring pattern to profile
        model: Optional model using the wiring (for performance profiling)
        batch_size: Batch size for performance testing
        seq_length: Sequence length for performance testing
        input_size: Input size for performance testing
        num_runs: Number of runs for averaging
    
    Returns:
        Tuple of (performance_stats, connectivity_stats)
    """
    profiler = WiringProfiler(wiring)
    
    # Get connectivity stats
    connectivity_stats = profiler.analyze_connectivity()
    
    # Get performance stats if model provided
    performance_stats: Dict[str, float] = {}
    if model is not None:
        fwd_stats = profiler.profile_forward(
            model,
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size,
            num_runs=num_runs
        )
        bwd_stats = profiler.profile_backward(
            model,
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size,
            num_runs=num_runs
        )
        performance_stats = {
            'forward_time': fwd_stats['mean'],
            'backward_time': bwd_stats['mean'],
            'memory_mb': profiler._measure_memory()
        }
    
    return performance_stats, connectivity_stats
