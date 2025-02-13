"""Visualization tools for neural circuit policies."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import mlx.core as mx


class WiringVisualizer:
    """Visualize wiring patterns and performance metrics."""
    
    def __init__(self, wiring: 'Wiring'):  # type: ignore
        self.wiring = wiring
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from wiring pattern."""
        self.graph = nx.DiGraph()
        
        # Add nodes
        for i in range(self.wiring.units):
            self.graph.add_node(i)
        
        # Add edges from adjacency matrix
        adj = self.wiring.adjacency_matrix
        for i in range(self.wiring.units):
            for j in range(self.wiring.units):
                if adj[i, j] != 0:
                    self.graph.add_edge(i, j, weight=float(adj[i, j]))
    
    def plot_wiring(
        self,
        figsize: Tuple[int, int] = (10, 10),
        node_size: int = 100,
        node_color: str = '#1f77b4',
        edge_color: str = '#aaaaaa',
        with_labels: bool = True,
        layout: str = 'spring'
    ):
        """Plot wiring pattern.
        
        Args:
            figsize: Figure size
            node_size: Size of nodes
            node_color: Color of nodes
            edge_color: Color of edges
            with_labels: Show node labels
            layout: Graph layout ('spring', 'circular', 'kamada_kawai', 'shell')
        """
        plt.figure(figsize=figsize)
        
        # Get layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:  # shell
            pos = nx.shell_layout(self.graph)
        
        # Draw network
        nx.draw(
            self.graph,
            pos,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=with_labels,
            arrows=True,
            alpha=0.7
        )
        
        plt.title('Wiring Pattern')
        plt.show()
    
    def plot_connectivity_matrix(
        self,
        figsize: Tuple[int, int] = (8, 8),
        cmap: str = 'viridis'
    ):
        """Plot connectivity matrix.
        
        Args:
            figsize: Figure size
            cmap: Colormap for heatmap
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.wiring.adjacency_matrix, cmap=cmap)
        plt.colorbar(label='Weight')
        plt.title('Connectivity Matrix')
        plt.xlabel('To Node')
        plt.ylabel('From Node')
        plt.show()
    
    def plot_degree_distribution(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """Plot in/out degree distributions.
        
        Args:
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Get degrees
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        
        plt.subplot(121)
        plt.hist(in_degrees, bins=20, alpha=0.7)
        plt.title('In-Degree Distribution')
        plt.xlabel('In-Degree')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.subplot(122)
        plt.hist(out_degrees, bins=20, alpha=0.7)
        plt.title('Out-Degree Distribution')
        plt.xlabel('Out-Degree')
        plt.ylabel('Count')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_path_lengths(
        self,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot path length distribution.
        
        Args:
            figsize: Figure size
        """
        # Get all shortest paths
        path_lengths = []
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source != target:
                    try:
                        length = nx.shortest_path_length(
                            self.graph,
                            source=source,
                            target=target
                        )
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        continue
        
        plt.figure(figsize=figsize)
        plt.hist(path_lengths, bins=20, alpha=0.7)
        plt.title('Path Length Distribution')
        plt.xlabel('Path Length')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()


class PerformanceVisualizer:
    """Visualize performance metrics."""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'memory': [],
            'time': [],
            'tflops': []
        }
    
    def add_metrics(
        self,
        loss: Optional[float] = None,
        memory: Optional[float] = None,
        time: Optional[float] = None,
        tflops: Optional[float] = None
    ):
        """Add performance metrics.
        
        Args:
            loss: Training loss
            memory: Memory usage (MB)
            time: Execution time (s)
            tflops: Compute throughput
        """
        if loss is not None:
            self.history['loss'].append(loss)
        if memory is not None:
            self.history['memory'].append(memory)
        if time is not None:
            self.history['time'].append(time)
        if tflops is not None:
            self.history['tflops'].append(tflops)
    
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 5),
        rolling_window: int = 1
    ):
        """Plot performance metrics.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size
            rolling_window: Window size for moving average
        """
        if metrics is None:
            metrics = [k for k, v in self.history.items() if v]
        
        n_metrics = len(metrics)
        plt.figure(figsize=figsize)
        
        for i, metric in enumerate(metrics, 1):
            values = self.history[metric]
            if not values:
                continue
            
            plt.subplot(1, n_metrics, i)
            
            # Apply moving average
            if rolling_window > 1:
                values = np.convolve(
                    values,
                    np.ones(rolling_window)/rolling_window,
                    mode='valid'
                )
            
            plt.plot(values)
            plt.title(f'{metric.title()} History')
            plt.xlabel('Step')
            plt.ylabel(metric.title())
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation(
        self,
        metric1: str,
        metric2: str,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """Plot correlation between two metrics.
        
        Args:
            metric1: First metric
            metric2: Second metric
            figsize: Figure size
        """
        if not (self.history[metric1] and self.history[metric2]):
            return
        
        plt.figure(figsize=figsize)
        plt.scatter(
            self.history[metric1],
            self.history[metric2],
            alpha=0.5
        )
        plt.xlabel(metric1.title())
        plt.ylabel(metric2.title())
        plt.title(f'{metric1.title()} vs {metric2.title()}')
        plt.grid(True)
        plt.show()


class ProfileVisualizer:
    """Visualize profiling results."""
    
    def __init__(self, profiler: 'MLXProfiler'):  # type: ignore
        self.profiler = profiler
    
    def plot_compute_profile(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """Plot compute profiling results.
        
        Args:
            figsize: Figure size
        """
        history = self.profiler.history['compute']
        if not history:
            return
        
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        plt.plot([h['tflops'] for h in history], marker='o')
        plt.title('Compute Efficiency')
        plt.xlabel('Profile Run')
        plt.ylabel('TFLOPS')
        plt.grid(True)
        
        plt.subplot(122)
        plt.plot([h['time_mean']*1000 for h in history], marker='o')
        plt.fill_between(
            range(len(history)),
            [(h['time_mean'] - h['time_std'])*1000 for h in history],
            [(h['time_mean'] + h['time_std'])*1000 for h in history],
            alpha=0.3
        )
        plt.title('Execution Time')
        plt.xlabel('Profile Run')
        plt.ylabel('Time (ms)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_memory_profile(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """Plot memory profiling results.
        
        Args:
            figsize: Figure size
        """
        history = self.profiler.history['memory']
        if not history:
            return
        
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        plt.plot([h['peak_usage'] for h in history], marker='o')
        plt.title('Peak Memory Usage')
        plt.xlabel('Profile Run')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        
        plt.subplot(122)
        plt.plot([h['total_allocated'] for h in history], marker='o')
        plt.plot([h['total_freed'] for h in history], marker='o')
        plt.title('Memory Allocation/Free')
        plt.xlabel('Profile Run')
        plt.ylabel('Memory (MB)')
        plt.legend(['Allocated', 'Freed'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_stream_profile(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ):
        """Plot stream profiling results.
        
        Args:
            figsize: Figure size
        """
        history = self.profiler.history['stream']
        if not history:
            return
        
        plt.figure(figsize=figsize)
        
        plt.subplot(121)
        plt.plot([h['kernel_time']*1000 for h in history], marker='o')
        plt.plot([h['memory_time']*1000 for h in history], marker='o')
        plt.title('Operation Times')
        plt.xlabel('Profile Run')
        plt.ylabel('Time (ms)')
        plt.legend(['Kernel', 'Memory'])
        plt.grid(True)
        
        plt.subplot(122)
        plt.plot([h['num_kernels'] for h in history], marker='o')
        plt.plot([h['num_transfers'] for h in history], marker='o')
        plt.title('Operation Counts')
        plt.xlabel('Profile Run')
        plt.ylabel('Count')
        plt.legend(['Kernels', 'Transfers'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


def plot_comparison(
    results: Dict[str, Dict[str, Union[float, List[float]]]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """Plot comparison of different configurations.
    
    Args:
        results: Dictionary of results
        metrics: List of metrics to plot
        figsize: Figure size
    """
    if metrics is None:
        # Get all available metrics
        metrics = set()
        for result in results.values():
            metrics.update(result.keys())
        metrics = sorted(list(metrics))
    
    n_metrics = len(metrics)
    plt.figure(figsize=figsize)
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, n_metrics, i)
        
        for name, result in results.items():
            if metric not in result:
                continue
            
            value = result[metric]
            if isinstance(value, list):
                plt.plot(value, label=name)
            else:
                plt.bar(name, value)
        
        plt.title(f'{metric.title()}')
        if isinstance(next(iter(results.values()))[metric], list):
            plt.xlabel('Step')
            plt.legend()
        else:
            plt.xticks(rotation=45)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
