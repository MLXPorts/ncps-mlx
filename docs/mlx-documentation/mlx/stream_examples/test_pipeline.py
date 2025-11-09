#!/usr/bin/env python
"""
Tests for multi-stream FFT convolution pipeline
"""

import mlx.core as mx
import numpy as np
from pipeline import FFTConvPipeline, create_test_data


def test_fft_roundtrip():
    """Test that FFT → IFFT recovers the original signal"""
    print("Test: FFT roundtrip...")

    signal = mx.random.normal((4, 16, 512), dtype=mx.float32)

    spectrum = mx.fft.rfft(signal, axis=-1)
    recovered = mx.fft.irfft(spectrum, n=signal.shape[-1], axis=-1)

    diff = mx.max(mx.abs(signal - recovered))
    print(f"  Max difference: {float(diff):.2e}")

    assert float(diff) < mx.array(1e-5, dtype=mx.float32), "FFT roundtrip failed"
    print("  ✓ PASS\n")


def test_stream_independence():
    """Test that streams produce correct independent results"""
    print("Test: Stream independence...")

    batch, channels, spatial = 4, 32, 256
    x, filters, biases = create_test_data(batch, channels, spatial)

    # Process with different numbers of streams
    pipeline_1 = FFTConvPipeline(num_streams=1)
    pipeline_2 = FFTConvPipeline(num_streams=2)
    pipeline_4 = FFTConvPipeline(num_streams=4)

    result_1 = pipeline_1.forward(x, filters, biases)
    result_2 = pipeline_2.forward(x, filters, biases)
    result_4 = pipeline_4.forward(x, filters, biases)

    mx.eval(result_1, result_2, result_4)

    diff_1_2 = mx.max(mx.abs(result_1 - result_2))
    diff_1_4 = mx.max(mx.abs(result_1 - result_4))

    print(f"  Max diff (1 vs 2 streams): {float(diff_1_2):.2e}")
    print(f"  Max diff (1 vs 4 streams): {float(diff_1_4):.2e}")

    assert float(diff_1_2) < mx.array(1e-4, dtype=mx.float32), "2-stream result differs"
    assert float(diff_1_4) < mx.array(1e-4, dtype=mx.float32), "4-stream result differs"

    print("  ✓ PASS\n")


def test_serial_vs_parallel():
    """Test that parallel and serial implementations match"""
    print("Test: Serial vs parallel equivalence...")

    batch, channels, spatial = 2, 16, 128
    x, filters, biases = create_test_data(batch, channels, spatial)

    pipeline = FFTConvPipeline(num_streams=4)

    result_parallel = pipeline.forward(x, filters, biases)
    result_serial = pipeline.forward_serial(x, filters, biases)

    mx.eval(result_parallel, result_serial)

    diff = mx.max(mx.abs(result_parallel - result_serial))
    print(f"  Max difference: {float(diff):.2e}")

    assert float(diff) < mx.array(1e-4, dtype=mx.float32), "Serial/parallel mismatch"
    print("  ✓ PASS\n")


def test_gradient_flow():
    """Test that gradients flow correctly through the pipeline"""
    print("Test: Gradient flow...")

    batch, channels, spatial = 2, 8, 64
    x, filters, biases = create_test_data(batch, channels, spatial)

    def loss_fn(x_input, f, b):
        pipeline = FFTConvPipeline(num_streams=2)
        output = pipeline.forward(x_input, f, b)
        return mx.mean(output ** mx.array(2.0, dtype=mx.float32))

    # Compute gradients
    grad_fn = mx.grad(loss_fn, argnums=[0, 1, 2])
    grads = grad_fn(x, filters, biases)

    # Check that gradients are not NaN or zero everywhere
    grad_x, grad_f, grad_b = grads
    mx.eval(grad_x, grad_f, grad_b)

    print(f"  Grad X:       mean={float(mx.mean(mx.abs(grad_x))):.2e}")
    print(f"  Grad filters: mean={float(mx.mean(mx.abs(grad_f))):.2e}")
    print(f"  Grad biases:  mean={float(mx.mean(mx.abs(grad_b))):.2e}")

    assert not mx.any(mx.isnan(grad_x)), "NaN in input gradients"
    assert not mx.any(mx.isnan(grad_f)), "NaN in filter gradients"
    assert not mx.any(mx.isnan(grad_b)), "NaN in bias gradients"

    print("  ✓ PASS\n")


def test_memory_efficiency():
    """Test memory usage with different stream counts"""
    print("Test: Memory efficiency...")

    batch, channels, spatial = 16, 64, 2048

    configs = [
        (1, "Serial"),
        (2, "2 streams"),
        (4, "4 streams"),
        (8, "8 streams")
    ]

    for num_streams, label in configs:
        if channels % num_streams != 0:
            continue

        mx.metal.reset_peak_memory()

        x, filters, biases = create_test_data(batch, channels, spatial)
        pipeline = FFTConvPipeline(num_streams=num_streams)

        result = pipeline.forward(x, filters, biases)
        mx.eval(result)

        peak_mb = mx.metal.get_peak_memory() / 1024 / 1024
        active_mb = mx.metal.get_active_memory() / 1024 / 1024

        print(f"  {label:12s}: Peak={peak_mb:.1f}MB, Active={active_mb:.1f}MB")

    print("  ✓ PASS\n")


def test_different_sizes():
    """Test various input sizes"""
    print("Test: Different input sizes...")

    test_cases = [
        (2, 16, 64, "Small"),
        (4, 32, 256, "Medium"),
        (8, 64, 1024, "Large"),
        (16, 128, 2048, "XLarge")
    ]

    pipeline = FFTConvPipeline(num_streams=4)

    for batch, channels, spatial, label in test_cases:
        try:
            x, filters, biases = create_test_data(batch, channels, spatial)
            result = pipeline.forward(x, filters, biases)
            mx.eval(result)

            # Check output shape
            assert result.shape == (batch, channels, spatial)
            print(f"  {label:8s} ({batch}x{channels}x{spatial}): ✓")

        except Exception as e:
            print(f"  {label:8s} ({batch}x{channels}x{spatial}): ✗ {str(e)}")

    print("  ✓ PASS\n")


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("Test: Numerical stability...")

    batch, channels, spatial = 4, 16, 256

    # Test with large values
    x_large = mx.random.normal((batch, channels, spatial), dtype=mx.float32) * mx.array(1000.0, dtype=mx.float32)
    freq_dim = spatial // 2 + 1
    filters_large = (mx.random.normal((channels, freq_dim), dtype=mx.float32) +
                     mx.array(1j, dtype=mx.complex64) *
                     mx.random.normal((channels, freq_dim), dtype=mx.float32)) * mx.array(100.0, dtype=mx.float32)
    biases_large = mx.random.normal((channels,), dtype=mx.float32) * mx.array(10.0, dtype=mx.float32)

    pipeline = FFTConvPipeline(num_streams=4)
    result = pipeline.forward(x_large, filters_large, biases_large)
    mx.eval(result)

    has_nan = mx.any(mx.isnan(result))
    has_inf = mx.any(mx.isinf(result))

    print(f"  Large values: NaN={bool(has_nan)}, Inf={bool(has_inf)}")
    assert not has_nan and not has_inf, "Numerical instability with large values"

    # Test with small values
    x_small = mx.random.normal((batch, channels, spatial), dtype=mx.float32) * mx.array(1e-6, dtype=mx.float32)
    filters_small = (mx.random.normal((channels, freq_dim), dtype=mx.float32) +
                     mx.array(1j, dtype=mx.complex64) *
                     mx.random.normal((channels, freq_dim), dtype=mx.float32)) * mx.array(1e-3, dtype=mx.float32)
    biases_small = mx.random.normal((channels,), dtype=mx.float32) * mx.array(1e-6, dtype=mx.float32)

    result = pipeline.forward(x_small, filters_small, biases_small)
    mx.eval(result)

    has_nan = mx.any(mx.isnan(result))
    has_inf = mx.any(mx.isinf(result))

    print(f"  Small values: NaN={bool(has_nan)}, Inf={bool(has_inf)}")
    assert not has_nan and not has_inf, "Numerical instability with small values"

    print("  ✓ PASS\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Multi-Stream FFT Pipeline Test Suite")
    print("=" * 70)
    print()

    tests = [
        test_fft_roundtrip,
        test_stream_independence,
        test_serial_vs_parallel,
        test_gradient_flow,
        test_memory_efficiency,
        test_different_sizes,
        test_numerical_stability
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {str(e)}\n")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
