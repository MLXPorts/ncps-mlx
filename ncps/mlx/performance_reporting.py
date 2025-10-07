"""Performance reporting tools for MLX Neural Circuit Policies."""

import json
import time
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt

from ncps.tests.configs.device_configs import get_device_config

class PerformanceReport:
    """Generate and manage performance reports."""
    
    def __init__(
        self,
        device_type: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        self.device_config = get_device_config(device_type)
        self.save_path = save_path
        self.metrics = {
            'device': self.device_config.device_type,
            'timestamp': time.time(),
            'tests': {},
            'summary': {}
        }
    
    def add_test_result(
        self,
        test_name: str,
        metrics: Dict[str, Union[float, int, str]]
    ):
        """Add test result to report."""
        self.metrics['tests'][test_name] = {
            'timestamp': time.time(),
            **metrics
        }
    
    def add_summary_metrics(
        self,
        metrics: Dict[str, Union[float, int, str]]
    ):
        """Add summary metrics to report."""
        self.metrics['summary'].update(metrics)
    
    def validate_performance(self) -> Dict[str, bool]:
        """Validate performance against device requirements."""
        return {
            'tflops': (
                self.metrics['summary'].get('tflops', 0) >=
                self.device_config.min_tflops
            ),
            'bandwidth': (
                self.metrics['summary'].get('bandwidth', 0) >=
                self.device_config.min_bandwidth
            ),
            'memory': (
                self.metrics['summary'].get('peak_memory', float('inf')) <=
                self.device_config.memory_budget
            )
        }
    
    def save(self, path: Optional[str] = None):
        """Save report to file."""
        save_path = path or self.save_path
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
    
    def plot_performance(self, show: bool = True):
        """Create performance visualization."""
        fig = plt.figure(figsize=(15, 10))
        
        # Plot TFLOPS
        plt.subplot(221)
        if 'neural_engine' in self.metrics['tests']:
            ne_test = self.metrics['tests']['neural_engine']
            plt.plot(
                ne_test.get('batch_sizes', []),
                ne_test.get('tflops', []),
                marker='o'
            )
            plt.axhline(
                y=self.device_config.min_tflops,
                color='r',
                linestyle='--',
                label='Minimum Required'
            )
        plt.xlabel('Batch Size')
        plt.ylabel('TFLOPS')
        plt.title('Neural Engine Performance')
        plt.grid(True)
        plt.legend()
        
        # Plot Memory Usage
        plt.subplot(222)
        if 'memory' in self.metrics['tests']:
            mem_test = self.metrics['tests']['memory']
            plt.plot(
                mem_test.get('batch_sizes', []),
                mem_test.get('memory_usage', []),
                marker='o'
            )
            plt.axhline(
                y=self.device_config.memory_budget,
                color='r',
                linestyle='--',
                label='Memory Budget'
            )
        plt.xlabel('Batch Size')
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage')
        plt.grid(True)
        plt.legend()
        
        # Plot Bandwidth
        plt.subplot(223)
        if 'memory' in self.metrics['tests']:
            mem_test = self.metrics['tests']['memory']
            plt.plot(
                mem_test.get('batch_sizes', []),
                mem_test.get('bandwidth', []),
                marker='o'
            )
            plt.axhline(
                y=self.device_config.min_bandwidth,
                color='r',
                linestyle='--',
                label='Minimum Required'
            )
        plt.xlabel('Batch Size')
        plt.ylabel('Bandwidth (GB/s)')
        plt.title('Memory Bandwidth')
        plt.grid(True)
        plt.legend()
        
        # Plot Compilation Speedup
        plt.subplot(224)
        if 'compilation' in self.metrics['tests']:
            comp_test = self.metrics['tests']['compilation']
            plt.bar(
                ['Uncompiled', 'Compiled'],
                [1.0, comp_test.get('speedup', 1.0)]
            )
        plt.ylabel('Relative Speed')
        plt.title('Compilation Speedup')
        plt.grid(True)
        
        plt.tight_layout()
        if show:
            plt.show()
        return fig
    
    def generate_markdown(self) -> str:
        """Generate markdown report."""
        validation = self.validate_performance()
        
        md = [
            f"# Performance Report for {self.device_config.device_type}",
            "",
            "## Summary",
            "",
            "| Metric | Value | Requirement | Status |",
            "|--------|--------|-------------|---------|"
        ]
        
        # Add summary metrics
        if 'tflops' in self.metrics['summary']:
            md.append(
                f"| TFLOPS | {self.metrics['summary']['tflops']:.2f} | "
                f">= {self.device_config.min_tflops:.2f} | "
                f"{'✓' if validation['tflops'] else '✗'} |"
            )
        
        if 'bandwidth' in self.metrics['summary']:
            md.append(
                f"| Bandwidth | {self.metrics['summary']['bandwidth']:.2f} GB/s | "
                f">= {self.device_config.min_bandwidth:.2f} GB/s | "
                f"{'✓' if validation['bandwidth'] else '✗'} |"
            )
        
        if 'peak_memory' in self.metrics['summary']:
            md.append(
                f"| Memory | {self.metrics['summary']['peak_memory']:.2f} MB | "
                f"<= {self.device_config.memory_budget:.2f} MB | "
                f"{'✓' if validation['memory'] else '✗'} |"
            )
        
        # Add test details
        md.extend([
            "",
            "## Test Details",
            ""
        ])
        
        for test_name, results in self.metrics['tests'].items():
            md.extend([
                f"### {test_name}",
                ""
            ])
            
            # Add test-specific metrics
            for metric, value in results.items():
                if metric != 'timestamp':
                    if isinstance(value, (int, float)):
                        md.append(f"- {metric}: {value:.2f}")
                    else:
                        md.append(f"- {metric}: {value}")
            md.append("")
        
        return "\n".join(md)
    
    @classmethod
    def load(cls, path: str) -> 'PerformanceReport':
        """Load report from file."""
        with open(path, 'r') as f:
            metrics = json.load(f)
        
        report = cls(device_type=metrics['device'])
        report.metrics = metrics
        return report

def compare_reports(reports: List[PerformanceReport], show: bool = True):
    """Compare multiple performance reports."""
    fig = plt.figure(figsize=(15, 10))
    
    # Compare TFLOPS
    plt.subplot(221)
    for report in reports:
        if 'tflops' in report.metrics['summary']:
            plt.bar(
                report.device_config.device_type,
                report.metrics['summary']['tflops']
            )
    plt.ylabel('TFLOPS')
    plt.title('Neural Engine Performance')
    plt.grid(True)
    
    # Compare Memory Usage
    plt.subplot(222)
    for report in reports:
        if 'peak_memory' in report.metrics['summary']:
            plt.bar(
                report.device_config.device_type,
                report.metrics['summary']['peak_memory']
            )
    plt.ylabel('Memory (MB)')
    plt.title('Peak Memory Usage')
    plt.grid(True)
    
    # Compare Bandwidth
    plt.subplot(223)
    for report in reports:
        if 'bandwidth' in report.metrics['summary']:
            plt.bar(
                report.device_config.device_type,
                report.metrics['summary']['bandwidth']
            )
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Memory Bandwidth')
    plt.grid(True)
    
    # Compare Compilation Speedup
    plt.subplot(224)
    for report in reports:
        if 'compilation' in report.metrics['tests']:
            plt.bar(
                report.device_config.device_type,
                report.metrics['tests']['compilation'].get('speedup', 1.0)
            )
    plt.ylabel('Compilation Speedup')
    plt.title('Compilation Effect')
    plt.grid(True)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig