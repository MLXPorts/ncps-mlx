#!/usr/bin/env python3
"""Test runner for MLX Neural Circuit Policy tests."""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional
import unittest

from ncps.tests.configs.device_configs import get_device_config
from ncps.mlx.advanced_profiling import MLXProfiler

def setup_args() -> argparse.Namespace:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Run MLX Neural Circuit Policy tests')
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['M1', 'M1 Pro', 'M1 Max', 'M1 Ultra'],
        help='Target device type'
    )
    
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['unit', 'performance', 'all'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of runs for performance tests'
    )
    
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save performance report'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def run_unit_tests(device_config: Dict, verbose: bool = False) -> bool:
    """Run unit tests."""
    # Set environment variable for device type
    os.environ['DEVICE_TYPE'] = device_config.device_type
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.discover('ncps/tests', pattern='test_mlx*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_performance_tests(
    device_config: Dict,
    num_runs: int,
    verbose: bool = False
) -> Dict:
    """Run performance tests."""
    from ncps.tests.mlx_benchmarks import AppleSiliconBenchmarks
    
    # Create and run benchmarks
    benchmarks = AppleSiliconBenchmarks()
    benchmarks.setUp()
    
    results = {
        'device_type': device_config.device_type,
        'num_runs': num_runs,
        'timestamp': time.time(),
        'tests': {}
    }
    
    # Run Neural Engine tests
    if verbose:
        print("Running Neural Engine tests...")
    
    ne_results = benchmarks.test_neural_engine_performance()
    results['tests']['neural_engine'] = ne_results
    
    # Run memory tests
    if verbose:
        print("Running memory tests...")
    
    mem_results = benchmarks.test_memory_scaling()
    results['tests']['memory'] = mem_results
    
    # Run batch size tests
    if verbose:
        print("Running batch size tests...")
    
    batch_results = benchmarks.test_batch_size_scaling()
    results['tests']['batch_size'] = batch_results
    
    # Run compilation tests
    if verbose:
        print("Running compilation tests...")
    
    compile_results = benchmarks.test_compilation_effects()
    results['tests']['compilation'] = compile_results
    
    # Aggregate results
    results['summary'] = {
        'tflops': max(ne_results['tflops']),
        'bandwidth': max(mem_results['bandwidth']),
        'peak_memory': max(mem_results['peak_memory']),
        'compilation_speedup': compile_results['speedup']
    }
    
    return results

def save_report(results: Dict, path: str):
    """Save performance report to file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    """Main entry point."""
    args = setup_args()
    
    # Get device configuration
    device_config = get_device_config(args.device)
    
    if args.verbose:
        print(f"Running tests for {device_config.device_type}")
    
    success = True
    results = {}
    
    # Run unit tests
    if args.test_type in ['unit', 'all']:
        if args.verbose:
            print("\nRunning unit tests...")
        
        success &= run_unit_tests(device_config, args.verbose)
    
    # Run performance tests
    if args.test_type in ['performance', 'all']:
        if args.verbose:
            print("\nRunning performance tests...")
        
        results = run_performance_tests(
            device_config,
            args.runs,
            args.verbose
        )
        
        # Verify performance requirements
        tflops_ok = results['summary']['tflops'] >= device_config.min_tflops
        bandwidth_ok = results['summary']['bandwidth'] >= device_config.min_bandwidth
        memory_ok = results['summary']['peak_memory'] <= device_config.memory_budget
        
        success &= all([tflops_ok, bandwidth_ok, memory_ok])
        
        if args.verbose:
            print("\nPerformance Requirements:")
            print(f"TFLOPS: {results['summary']['tflops']:.2f} >= {device_config.min_tflops} : {'✓' if tflops_ok else '✗'}")
            print(f"Bandwidth: {results['summary']['bandwidth']:.2f} >= {device_config.min_bandwidth} : {'✓' if bandwidth_ok else '✗'}")
            print(f"Memory: {results['summary']['peak_memory']:.2f} <= {device_config.memory_budget} : {'✓' if memory_ok else '✗'}")
    
    # Save report if requested
    if args.report and results:
        save_report(results, args.report)
        if args.verbose:
            print(f"\nPerformance report saved to {args.report}")
    
    # Exit with appropriate status
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()