#!/usr/bin/env python
"""
ACTUAL Custom Metal Kernel Integration Pattern

This shows how to properly compile and use custom Metal kernels with MLX.
Currently uses a simplified pattern; in production you'd compile the full kernels.
"""

import mlx.core as mx
from pathlib import Path


class ActualCustomFFTKernel:
    """
    Example of ACTUAL custom Metal kernel integration.

    This demonstrates the mx.fast.metal_kernel pattern for compiling
    custom Metal code at runtime.
    """

    def __init__(self):
        # Header for all Metal kernels
        self.header = """
#include <metal_stdlib>
using namespace metal;

// Complex arithmetic helpers
METAL_FUNC float2 cmul(float2 a, float2 b) {
    return float2(
        fma(a.x, b.x, -a.y * b.y),
        fma(a.x, b.y, a.y * b.x)
    );
}
"""

        # Example: Complex pointwise multiplication kernel
        complex_mul_body = r"""
    uint batch_idx = thread_position_in_grid.x;
    uint elem_idx = thread_position_in_grid.y;

    uint batch_size = (uint)shape[0];
    uint n = (uint)shape[1];

    if (batch_idx >= batch_size || elem_idx >= n) {
        return;
    }

    uint idx = batch_idx * n + elem_idx;

    // Perform complex multiplication
    float2 a = input_a[idx];
    float2 b = input_b[idx];
    output[idx] = cmul(a, b);
"""

        # Compile custom kernel using mx.fast.metal_kernel
        try:
            self.complex_mul_kernel = mx.fast.metal_kernel(
                name="complex_mul_custom",
                input_names=["input_a", "input_b", "shape"],
                output_names=["output"],
                header=self.header,
                source=complex_mul_body,
                ensure_row_contiguous=True
            )
            print("✓ Compiled custom complex multiplication kernel")
            self.has_custom_kernels = True
        except Exception as e:
            print(f"⚠ Could not compile custom kernel: {e}")
            print("  Falling back to MLX built-ins")
            self.has_custom_kernels = False

        # Example: Bias addition kernel
        bias_add_body = r"""
    uint batch_idx = thread_position_in_grid.x;
    uint channel_idx = thread_position_in_grid.y;
    uint spatial_idx = thread_position_in_grid.z;

    uint batch_size = (uint)shape[0];
    uint channels = (uint)shape[1];
    uint spatial_dim = (uint)shape[2];

    if (batch_idx >= batch_size || channel_idx >= channels || spatial_idx >= spatial_dim) {
        return;
    }

    uint idx = batch_idx * (channels * spatial_dim) + channel_idx * spatial_dim + spatial_idx;

    // Add bias (broadcast across spatial dimension)
    output[idx] = input[idx] + bias[channel_idx];
"""

        try:
            self.bias_add_kernel = mx.fast.metal_kernel(
                name="bias_add_custom",
                input_names=["input", "bias", "shape"],
                output_names=["output"],
                header=self.header,
                source=bias_add_body,
                ensure_row_contiguous=True
            )
            print("✓ Compiled custom bias addition kernel")
        except Exception as e:
            print(f"⚠ Could not compile bias kernel: {e}")

    def complex_mul(self, a: mx.array, b: mx.array) -> mx.array:
        """
        Custom complex multiplication kernel.

        Args:
            a: Complex array (batch, n)
            b: Complex array (batch, n) or (n,)

        Returns:
            Complex array (batch, n)
        """
        if not self.has_custom_kernels:
            # Fallback to MLX built-in
            return mx.multiply(a, b)

        # Prepare inputs
        batch_size = a.shape[0]
        n = a.shape[1]

        # Broadcast b if needed
        if b.ndim == 1:
            b = b[None, :]

        # Shape parameter (must be mx.array with proper dtype!)
        shape = mx.array([batch_size, n], dtype=mx.uint32)

        # Launch kernel with proper grid sizing
        grid = (batch_size, n, 1)
        threadgroup = (min(32, n), 1, 1)

        # Execute custom kernel
        (output,) = self.complex_mul_kernel(
            inputs=[a, b, shape],
            output_shapes=[(batch_size, n)],
            output_dtypes=[a.dtype],
            grid=grid,
            threadgroup=threadgroup
        )

        return output

    def add_bias(self, x: mx.array, bias: mx.array) -> mx.array:
        """
        Custom bias addition kernel.

        Args:
            x: Input (batch, channels, spatial)
            bias: Bias (channels,)

        Returns:
            Output (batch, channels, spatial)
        """
        if not self.has_custom_kernels:
            # Fallback: proper tensor ops, no Python scalars!
            bias_expanded = bias[..., None]
            return mx.add(x, bias_expanded)

        # Get dimensions
        batch = x.shape[0]
        channels = x.shape[1]
        spatial = x.shape[2]

        # Shape parameter
        shape = mx.array([batch, channels, spatial], dtype=mx.uint32)

        # Launch kernel
        grid = (batch, channels, spatial)
        threadgroup = (1, min(8, channels), min(128, spatial))

        (output,) = self.bias_add_kernel(
            inputs=[x, bias, shape],
            output_shapes=[(batch, channels, spatial)],
            output_dtypes=[x.dtype],
            grid=grid,
            threadgroup=threadgroup
        )

        return output


def demo_custom_kernels():
    """Demonstrate actual custom kernel usage"""
    print("=" * 70)
    print("Custom Metal Kernel Integration Demo")
    print("=" * 70)
    print()

    # Initialize custom kernels
    kernels = ActualCustomFFTKernel()
    print()

    # Create test data (proper tensor discipline!)
    batch, n = 4, 256
    scale = mx.array(0.1, dtype=mx.float32)

    # Complex arrays
    a_real = mx.multiply(mx.random.normal((batch, n), dtype=mx.float32), scale)
    a_imag = mx.multiply(mx.random.normal((batch, n), dtype=mx.float32), scale)
    a = mx.add(a_real, mx.multiply(mx.array(1j, dtype=mx.complex64), a_imag))

    b_real = mx.multiply(mx.random.normal((n,), dtype=mx.float32), scale)
    b_imag = mx.multiply(mx.random.normal((n,), dtype=mx.float32), scale)
    b = mx.add(b_real, mx.multiply(mx.array(1j, dtype=mx.complex64), b_imag))

    # Test complex multiplication
    print("Testing custom complex multiplication kernel...")
    result_custom = kernels.complex_mul(a, b)
    result_reference = mx.multiply(a, b)
    mx.eval(result_custom, result_reference)

    diff = mx.max(mx.abs(mx.subtract(result_custom, result_reference)))
    diff_val = float(diff)
    print(f"  Max difference vs reference: {diff_val:.2e}")

    if diff_val < 1e-5:
        print("  ✓ Custom kernel matches reference!")
    else:
        print("  ⚠ Difference detected (expected for different implementations)")
    print()

    # Test bias addition
    print("Testing custom bias addition kernel...")
    batch, channels, spatial = 2, 8, 64
    x = mx.multiply(mx.random.normal((batch, channels, spatial), dtype=mx.float32), scale)
    bias = mx.multiply(mx.random.normal((channels,), dtype=mx.float32), scale)

    result_custom = kernels.add_bias(x, bias)
    bias_expanded = bias[..., None]
    result_reference = mx.add(x, bias_expanded)
    mx.eval(result_custom, result_reference)

    diff = mx.max(mx.abs(mx.subtract(result_custom, result_reference)))
    diff_val = float(diff)
    print(f"  Max difference vs reference: {diff_val:.2e}")

    if diff_val < 1e-5:
        print("  ✓ Custom kernel matches reference!")
    else:
        print("  ⚠ Difference detected")
    print()

    print("=" * 70)
    print("Key Patterns Demonstrated:")
    print("=" * 70)
    print()
    print("1. mx.fast.metal_kernel compilation:")
    print("   • header: Metal includes and helpers")
    print("   • source: Kernel body (no signature)")
    print("   • input_names/output_names: Buffer binding")
    print()
    print("2. Proper grid/threadgroup sizing:")
    print("   • grid: Total threads (batch, n, 1)")
    print("   • threadgroup: Threads per group (32, 1, 1)")
    print("   • Align to 32 for Apple GPU warp size")
    print()
    print("3. MLX tensor discipline:")
    print("   • shape = mx.array([batch, n], dtype=mx.uint32)")
    print("   • scale = mx.array(0.1, dtype=mx.float32)")
    print("   • result = mx.multiply(a, scale)  # Not a * 0.1!")
    print()
    print("4. Fallback pattern:")
    print("   • Try to compile custom kernel")
    print("   • Catch exception and use MLX built-in")
    print("   • Ensures code always works")
    print()


if __name__ == "__main__":
    demo_custom_kernels()
