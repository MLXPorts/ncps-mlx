"""Hardware-specific benchmarks for Neural Circuit Policies on Apple Silicon."""

import unittest
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple

from ncps.mlx import CfC, CfCCell, LTC, LTCCell
from ncps.wirings import AutoNCP
from ncps.mlx.advanced_profiling import MLXProfiler

class AppleSiliconBenchmarks(unittest.TestCase):
    """Benchmark suite for Apple Silicon devices."""
    
    @classmethod
    def setUpClass(cls):
        """Setup benchmark configurations."""
        cls.device_configs = {
            'M1': {
                'batch_sizes': [32, 64],
                'hidden_sizes': [64, 128],
                'backbone_units': [32, 64],
                'memory_budget': 8 * 1024  # 8GB in MB
            },
            'M1 Pro': {
                'batch_sizes': [64, 128],
                'hidden_sizes': [128, 256],
                'backbone_units': [64, 128],
                'memory_budget': 16 * 1024  # 16GB in MB
            },
            'M1 Max': {
                'batch_sizes': [128, 256],
                'hidden_sizes': [256, 512],
                'backbone_units': [128, 256],
                'memory_budget': 32 * 1024  # 32GB in MB
            },
            'M1 Ultra': {
                'batch_sizes': [256, 512],
                'hidden_sizes': [512, 1024],
                'backbone_units': [256, 512],
                'memory_budget': 64 * 1024  # 64GB in MB
            }
        }
        
        # Detect device type
        cls.device_type = cls.detect_device_type()
        cls.config = cls.device_configs[cls.device_type]
    
    @staticmethod
    def detect_device_type() -> str:
        """Detect Apple Silicon device type."""
        # This would be replaced with actual device detection
        # For now, return a default
        return 'M1'
    
    def create_model(self, hidden_size: int, backbone_units: List[int]) -> CfC:
        """Create model with specified configuration."""
        wiring = AutoNCP(
            units=hidden_size,
            output_size=hidden_size // 4
        )
        
        return CfC(
            cell=CfCCell(
                wiring=wiring,
                activation="tanh",
                backbone_units=backbone_units,
                backbone_layers=len(backbone_units)
            ),
            return_sequences=True
        )
    
    def benchmark_neural_engine(self, model: CfC, batch_size: int) -> Dict:
        """Benchmark Neural Engine performance."""
        profiler = MLXProfiler(model)
        
        # Create test data
        x = mx.random.normal((batch_size, 16, model.input_size))
        
        # Test without compilation
        stats_uncompiled = profiler.profile_compute(
            batch_size=batch_size,
            seq_length=16,
            num_runs=100
        )
        
        # Test with compilation
        @mx.compile(static_argnums=(1,))
        def forward(x, training=False):
            return model(x, training=training)
        
        stats_compiled = profiler.profile_compute(
            batch_size=batch_size,
            seq_length=16,
            num_runs=100,
            forward_fn=forward
        )
        
        return {
            'uncompiled_tflops': stats_uncompiled['tflops'],
            'compiled_tflops': stats_compiled['tflops'],
            'speedup': stats_uncompiled['time_mean'] / stats_compiled['time_mean'],
            'ne_utilization': stats_compiled['ne_utilization']
        }
    
    def benchmark_memory(self, model: CfC, batch_size: int) -> Dict:
        """Benchmark memory performance."""
        profiler = MLXProfiler(model)
        
        stats = profiler.profile_memory(
            batch_size=batch_size,
            track_unified=True
        )
        
        return {
            'peak_usage': stats['peak_usage'],
            'bandwidth': stats['bandwidth'],
            'utilization': stats['memory_utilization']
        }
    
    def test_neural_engine_performance(self):
        """Test Neural Engine performance scaling."""
        results = []
        
        for hidden_size in self.config['hidden_sizes']:
            for batch_size in self.config['batch_sizes']:
                model = self.create_model(
                    hidden_size=hidden_size,
                    backbone_units=[hidden_size, hidden_size]
                )
                
                ne_stats = self.benchmark_neural_engine(model, batch_size)
                mem_stats = self.benchmark_memory(model, batch_size)
                
                results.append({
                    'hidden_size': hidden_size,
                    'batch_size': batch_size,
                    **ne_stats,
                    **mem_stats
                })
                
                # Verify performance meets device expectations
                self.assertGreater(ne_stats['compiled_tflops'], 1.0)
                self.assertGreater(ne_stats['speedup'], 1.5)
                self.assertGreater(ne_stats['ne_utilization'], 50)
                self.assertLess(mem_stats['peak_usage'], self.config['memory_budget'])
    
    def test_memory_scaling(self):
        """Test memory usage scaling."""
        for hidden_size in self.config['hidden_sizes']:
            model = self.create_model(
                hidden_size=hidden_size,
                backbone_units=[hidden_size, hidden_size]
            )
            
            for batch_size in self.config['batch_sizes']:
                mem_stats = self.benchmark_memory(model, batch_size)
                
                # Verify memory usage is reasonable
                self.assertLess(
                    mem_stats['peak_usage'],
                    self.config['memory_budget']
                )
                self.assertGreater(mem_stats['bandwidth'], 50)  # GB/s
                self.assertGreater(mem_stats['utilization'], 70)  # %
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        hidden_size = self.config['hidden_sizes'][0]
        model = self.create_model(
            hidden_size=hidden_size,
            backbone_units=[hidden_size, hidden_size]
        )
        
        prev_tflops = 0
        for batch_size in sorted(self.config['batch_sizes']):
            ne_stats = self.benchmark_neural_engine(model, batch_size)
            
            # Verify performance increases with batch size
            self.assertGreater(ne_stats['compiled_tflops'], prev_tflops)
            prev_tflops = ne_stats['compiled_tflops']
    
    def test_compilation_effects(self):
        """Test effects of compilation."""
        hidden_size = self.config['hidden_sizes'][0]
        batch_size = self.config['batch_sizes'][0]
        
        model = self.create_model(
            hidden_size=hidden_size,
            backbone_units=[hidden_size, hidden_size]
        )
        
        ne_stats = self.benchmark_neural_engine(model, batch_size)
        
        # Verify compilation improves performance
        self.assertGreater(ne_stats['speedup'], 1.5)
        self.assertGreater(
            ne_stats['compiled_tflops'],
            ne_stats['uncompiled_tflops']
        )

if __name__ == '__main__':
    unittest.main()