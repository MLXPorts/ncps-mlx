// Simplified FFT kernel for power-of-2 sizes
// Optimized for the multi-stream pipeline use case

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// COMPLEX ARITHMETIC
// ============================================================================

METAL_FUNC float2 cmul(float2 a, float2 b) {
    return float2(
        fma(a.x, b.x, -a.y * b.y),
        fma(a.x, b.y, a.y * b.x)
    );
}

METAL_FUNC float2 cadd(float2 a, float2 b) {
    return a + b;
}

METAL_FUNC float2 csub(float2 a, float2 b) {
    return a - b;
}

METAL_FUNC float2 get_twiddle(int k, int n, bool inverse) {
    float sign = inverse ? 1.0f : -1.0f;
    float theta = sign * 2.0f * M_PI_F * float(k) / float(n);
    return float2(metal::fast::cos(theta), metal::fast::sin(theta));
}

// ============================================================================
// RADIX-2 BUTTERFLY
// ============================================================================

METAL_FUNC void radix2_butterfly(thread float2& a, thread float2& b) {
    float2 temp = a;
    a = cadd(temp, b);
    b = csub(temp, b);
}

// ============================================================================
// STOCKHAM FFT KERNEL (Power-of-2 only, in-threadgroup)
// ============================================================================

kernel void fft_1d_pow2(
    const device float2* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant bool& inverse [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tg_size [[threads_per_threadgroup]])
{
    // Each threadgroup processes one FFT
    // Shared memory for in-place computation
    threadgroup float2 shared[1024]; // Max 1024 complex elements

    uint batch_idx = gid.x;
    uint local_idx = tid.x;

    if (batch_idx >= batch_size || n > 1024) {
        return;
    }

    // Load data into shared memory
    if (local_idx < n) {
        shared[local_idx] = input[batch_idx * n + local_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stockham FFT algorithm
    uint log2n = 0;
    uint temp_n = n;
    while (temp_n > 1) {
        log2n++;
        temp_n >>= 1;
    }

    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1u << stage;        // Size of sub-FFTs
        uint m2 = m << 1;            // Size after combining

        if (local_idx < n) {
            uint k = local_idx & (m - 1);              // Index within sub-FFT
            uint j = ((local_idx >> stage) << (stage + 1)) + k;  // Bit-reversed index

            if ((local_idx & m) == 0) {
                // First half of butterfly
                float2 twiddle = get_twiddle(k, m2, inverse);
                float2 a = shared[j];
                float2 b = cmul(shared[j + m], twiddle);

                shared[j] = cadd(a, b);
                shared[j + m] = csub(a, b);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply scaling for inverse FFT
    if (inverse && local_idx < n) {
        shared[local_idx] = shared[local_idx] / float(n);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write result
    if (local_idx < n) {
        output[batch_idx * n + local_idx] = shared[local_idx];
    }
}

// ============================================================================
// REAL-TO-COMPLEX FFT (RFFT)
// ============================================================================

kernel void rfft_1d_pow2(
    const device float* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]])
{
    threadgroup float2 shared[1024];

    uint batch_idx = gid.x;
    uint local_idx = tid.x;

    if (batch_idx >= batch_size || n > 1024) {
        return;
    }

    // Load real data as complex (imaginary = 0)
    if (local_idx < n) {
        shared[local_idx] = float2(input[batch_idx * n + local_idx], 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Run complex FFT
    uint log2n = 0;
    uint temp_n = n;
    while (temp_n > 1) {
        log2n++;
        temp_n >>= 1;
    }

    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1u << stage;
        uint m2 = m << 1;

        if (local_idx < n) {
            uint k = local_idx & (m - 1);
            uint j = ((local_idx >> stage) << (stage + 1)) + k;

            if ((local_idx & m) == 0) {
                float2 twiddle = get_twiddle(k, m2, false);
                float2 a = shared[j];
                float2 b = cmul(shared[j + m], twiddle);

                shared[j] = cadd(a, b);
                shared[j + m] = csub(a, b);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write only positive frequencies (exploit Hermitian symmetry)
    uint out_size = n / 2 + 1;
    if (local_idx < out_size) {
        output[batch_idx * out_size + local_idx] = shared[local_idx];
    }
}

// ============================================================================
// COMPLEX POINTWISE MULTIPLICATION
// ============================================================================

kernel void complex_mul_pointwise(
    const device float2* input [[buffer(0)]],
    const device float2* filter [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.x;
    uint elem_idx = gid.y;

    if (batch_idx >= batch_size || elem_idx >= n) {
        return;
    }

    uint idx = batch_idx * n + elem_idx;
    output[idx] = cmul(input[idx], filter[elem_idx]);
}

// ============================================================================
// COMPLEX-TO-REAL IFFT (IRFFT)
// ============================================================================

kernel void irfft_1d_pow2(
    const device float2* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],        // Output size (even)
    constant uint& batch_size [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]])
{
    threadgroup float2 shared[1024];

    uint batch_idx = gid.x;
    uint local_idx = tid.x;

    if (batch_idx >= batch_size || n > 1024) {
        return;
    }

    uint in_size = n / 2 + 1;

    // Reconstruct full spectrum from Hermitian symmetry
    if (local_idx < in_size) {
        shared[local_idx] = input[batch_idx * in_size + local_idx];
    }

    if (local_idx > 0 && local_idx < n / 2) {
        // Hermitian symmetry: X[n-k] = conj(X[k])
        uint mirror_idx = n - local_idx;
        if (mirror_idx < n) {
            shared[mirror_idx] = float2(shared[local_idx].x, -shared[local_idx].y);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Run inverse complex FFT
    uint log2n = 0;
    uint temp_n = n;
    while (temp_n > 1) {
        log2n++;
        temp_n >>= 1;
    }

    for (uint stage = 0; stage < log2n; stage++) {
        uint m = 1u << stage;
        uint m2 = m << 1;

        if (local_idx < n) {
            uint k = local_idx & (m - 1);
            uint j = ((local_idx >> stage) << (stage + 1)) + k;

            if ((local_idx & m) == 0) {
                float2 twiddle = get_twiddle(k, m2, true);  // inverse=true
                float2 a = shared[j];
                float2 b = cmul(shared[j + m], twiddle);

                shared[j] = cadd(a, b);
                shared[j + m] = csub(a, b);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply scaling and take real part
    if (local_idx < n) {
        output[batch_idx * n + local_idx] = shared[local_idx].x / float(n);
    }
}

// ============================================================================
// BIAS ADDITION (Element-wise add)
// ============================================================================

kernel void add_bias(
    const device float* input [[buffer(0)]],
    const device float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.x;
    uint elem_idx = gid.y;

    if (batch_idx >= batch_size || elem_idx >= n) {
        return;
    }

    uint idx = batch_idx * n + elem_idx;
    output[idx] = input[idx] + bias[elem_idx];
}
