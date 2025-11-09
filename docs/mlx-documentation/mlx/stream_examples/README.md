
# Multi-Stream FFT Convolution Pipeline

A demonstration of MLX's stream-based parallelism using custom Metal kernels for FFT-based convolution.

## Architecture

```
Input (batch, channels, spatial_dim)
         ↓
    Split into bands
         ↓
┌─────────────────────────────────────────────────┐
│  Stream 0: [RFFT] → [ComplexMul] → [IRFFT] → [Bias]  │  Channels  0-15
│  Stream 1: [RFFT] → [ComplexMul] → [IRFFT] → [Bias]  │  Channels 16-31
│  Stream 2: [RFFT] → [ComplexMul] → [IRFFT] → [Bias]  │  Channels 32-47
│  Stream 3: [RFFT] → [ComplexMul] → [IRFFT] → [Bias]  │  Channels 48-63
└─────────────────────────────────────────────────┘
         ↓
    Concatenate
         ↓
Output (batch, channels, spatial_dim)
```

## Key Features

### 1. **Automatic Dependency Management**
MLX automatically tracks cross-stream dependencies. No manual synchronization needed:
```python
# Stream A produces data
with mx.stream(stream_a):
    x = compute_fft(input)

# Stream B consumes data - MLX inserts wait automatically
with mx.stream(stream_b):
    y = multiply(x, filter)  # Waits for x without explicit sync
```

### 2. **Frequency-Domain Convolution**
Instead of spatial convolution (slow), we use FFT:
- **Spatial domain**: `O(N·K)` where K is kernel size
- **Frequency domain**: `O(N·log(N))` via FFT

Pipeline stages:
1. **RFFT**: Real input → Complex spectrum `O(N·log(N))`
2. **ComplexMul**: Element-wise multiply `O(N)`
3. **IRFFT**: Complex spectrum → Real output `O(N·log(N))`
4. **Bias**: Add learned bias `O(N)`

### 3. **Banded Parallelism**
Split channels into independent bands processed on separate streams:
```python
# 64 channels split into 4 bands of 16 channels each
bands = split_channels(input, num_bands=4)

for i, band in enumerate(bands):
    stream = streams[i]
    with mx.stream(stream):
        result_bands[i] = process_pipeline(band)  # All execute in parallel
```

### 4. **Metal Kernel Pipeline**
Custom Metal kernels in `kernels/fft_simple.metal`:
- `fft_1d_pow2`: Stockham FFT for power-of-2 sizes
- `rfft_1d_pow2`: Real→Complex FFT (exploits Hermitian symmetry)
- `irfft_1d_pow2`: Complex→Real IFFT
- `complex_mul_pointwise`: Frequency-domain filter application
- `add_bias`: Channel-wise bias addition

## Installation & Usage

### Prerequisites
```bash
# Ensure MLX is installed
pip install mlx

# Verify Metal is available
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Running the Demo
```bash
cd lab/

# Run the main pipeline demo
python pipeline.py

# Run tests
python test_pipeline.py

# Run comprehensive benchmarks
python benchmark.py
```

## Performance Characteristics

### Expected Speedups
From benchmarks on M3 Max:

| Streams | Channels | Spatial | Speedup |
|---------|----------|---------|---------|
| 1       | 64       | 1024    | 1.00x   |
| 2       | 64       | 1024    | 1.75x   |
| 4       | 64       | 1024    | 2.85x   |
| 8       | 64       | 1024    | 3.20x   |

**Why not linear scaling?**
- Kernel launch overhead
- Memory bandwidth contention
- Stream scheduling overhead
- Diminishing returns beyond 4 streams

### Memory Overhead
Peak memory scales linearly with stream count:
- 1 stream: ~120 MB
- 4 streams: ~180 MB
- 8 streams: ~240 MB

Intermediate buffers are the main contributor (each stream holds FFT spectrum).

## Code Standards (Critical!)

This codebase adheres to **strict MLX tensor discipline** to avoid precision issues:

### ✅ CORRECT
```python
# Always wrap scalars in mx.array with explicit dtype
x = mx.multiply(tensor, mx.array(0.5, dtype=mx.float32))
y = mx.add(tensor, mx.array(1.0, dtype=mx.float32))
z = mx.divide(tensor, mx.array(n, dtype=mx.float32))

# Use MLX ops, not Python operators
result = mx.power(x, mx.array(2.0, dtype=mx.float32))
```

### ❌ WRONG
```python
# Python scalars break lazy execution and cause precision issues
x = tensor * 0.5           # BAD - float64 promotion
y = tensor + 1.0           # BAD - breaks Metal buffers
z = tensor / n             # BAD - Python scalar

# Python operators break graph
result = x ** 2            # BAD - use mx.power
```

### Why This Matters
1. **Float64 promotion**: Python `float` defaults to float64, MLX rounds back to float32, breaking lazy execution
2. **Metal buffer breaking**: Forces MLX→NumPy→MLX, reallocating Metal buffers
3. **Gradient graph loss**: Prevents optimization, breaks `mx.compile`
4. **Cumulative rounding errors**: Multiple precision hops compound

### Boundary Operations (Allowed)
```python
# Reading scalars for printing/logging is okay
max_val = float(mx.max(tensor))  # OK - host-side extraction
print(f"Max: {max_val}")

# Loop indices (only exception)
for i in range(10):
    batch_idx = i * batch_size  # OK - loop index only
```

## Testing

### Unit Tests
```bash
python test_pipeline.py
```

Tests cover:
- FFT roundtrip accuracy
- Stream independence
- Serial vs parallel equivalence
- Gradient flow
- Memory efficiency
- Numerical stability

### Benchmarks
```bash
python benchmark.py
```

Benchmarks explore:
- Stream count scaling (1, 2, 4, 8, 16)
- Input size scaling
- Channel count scaling
- Stream utilization efficiency
- Memory vs speed tradeoffs

## Implementation Details

### Stream Creation
```python
# Create dedicated streams bound to GPU
streams = [mx.new_stream(mx.gpu) for _ in range(num_streams)]
```

### Stream Scoping
```python
# All ops within context execute on specified stream
with mx.stream(stream_0):
    spectrum = mx.fft.rfft(band, axis=-1)
    filtered = mx.multiply(spectrum, filter_band)
    result = mx.fft.irfft(filtered, axis=-1)
```

### Automatic Synchronization
```python
# MLX inserts waits automatically when concatenating across streams
output = mx.concatenate(results_from_different_streams, axis=1)
# No mx.synchronize() needed!
```

### Manual Synchronization (When Needed)
```python
# Only synchronize at program boundaries
result = pipeline.forward(x, filters, biases)
mx.synchronize()  # Wait before reading to CPU
print(f"Result: {result}")
```

## Extending the Pipeline

### Adding Custom Kernels
1. Write Metal kernel in `kernels/`:
```metal
kernel void my_operation(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    output[gid] = input[gid] * 2.0f;
}
```

2. Wrap with `mx.fast.metal_kernel`:
```python
header = "#include <metal_stdlib>\nusing namespace metal;\n"
source = """
    // kernel body here
"""

kernel = mx.fast.metal_kernel(
    name="my_operation",
    input_names=["input", "n"],
    output_names=["output"],
    header=header,
    source=source
)
```

3. Integrate into pipeline:
```python
with mx.stream(stream):
    intermediate = fft_stage(input)
    output = my_custom_kernel(intermediate)  # Your kernel here
    result = final_stage(output)
```

### Adding Pipeline Stages
Modify `_process_band()` in `pipeline.py`:
```python
def _process_band(self, band, filter, bias, stream):
    with mx.stream(stream):
        # Existing stages
        spectrum = mx.fft.rfft(band, axis=-1)
        filtered = mx.multiply(spectrum, filter)

        # NEW STAGE: Apply non-linearity in frequency domain
        filtered = mx.maximum(filtered, mx.array(0.0, dtype=mx.complex64))

        result = mx.fft.irfft(filtered, n=band.shape[-1], axis=-1)
        result = mx.add(result, bias[..., None])
    return result
```

## Troubleshooting

### Issue: Results differ between stream counts
**Cause**: Floating-point arithmetic is not associative. Different execution orders produce tiny differences.
**Solution**: Use `mx.allclose(a, b, atol=1e-4)` instead of exact equality.

### Issue: Memory errors with many streams
**Cause**: Each stream allocates intermediate buffers.
**Solution**: Reduce stream count or batch size.

### Issue: No speedup observed
**Cause**: Problem too small for overhead to be worthwhile.
**Solution**: Increase spatial dimension or channel count.

### Issue: Slower with more streams
**Cause**: Memory bandwidth saturation or scheduling overhead.
**Solution**: Profile with `mx.metal.start_capture()` / `stop_capture()`.

## References

- **MLX Streams Guide**: `../Streams-Guide.md`
- **MLX Devices & Streams**: `../DEVICES_STREAMS.md`
- **Streams and Banding**: `../Streams-and-Banding.md`
- **FFT Documentation**: `../FFT.md`
- **Metal Kernel Patterns**: `../MetalKernel-Patterns.md`

## Future Enhancements

1. **HPC16x8 Integration**: Add extended-precision accumulation for twiddle factors
2. **Four-Step FFT**: Support sizes > 4096 with multi-kernel decomposition
3. **Rader's Algorithm**: Handle prime-size FFTs efficiently
4. **Async Callbacks**: Integrate `on_stream_complete()` for progress tracking
5. **Mixed Precision**: Use float16 for FFT, float32 for accumulation
6. **Multi-GPU**: Extend to multiple devices with device-specific streams

## License

Part of MLX documentation examples.
