import mlx.core as mx

from ncps.mini_keras.backend import standardize_dtype
from ncps.mini_keras.backend.common import dtypes
from ncps.mini_keras.backend.mlx.core import convert_to_tensor


def _segment_reduction_fn(
    data, segment_ids, reduction_method, num_segments, sorted
):
    if num_segments is None:
        num_segments = mx.max(segment_ids) + 1

    valid_indices = segment_ids >= 0  # Ignore segment_ids that are -1
    valid_data = data[valid_indices]
    valid_segment_ids = segment_ids[valid_indices]

    data_shape = list(valid_data.shape)
    data_shape[0] = num_segments

    if reduction_method == mx.maximum:
        result = mx.ones(data_shape, dtype=valid_data.dtype) * float('-inf')
    else:
        result = mx.zeros(data_shape, dtype=valid_data.dtype)

    if sorted:
        # Note: MLX doesn't have direct equivalent to np.add.at
        # This needs custom implementation
        raise NotImplementedError("Sorted segment reduction not implemented for MLX")
    else:
        sort_indices = mx.argsort(valid_segment_ids)
        sorted_segment_ids = valid_segment_ids[sort_indices]
        sorted_data = valid_data[sort_indices]
        
        # Note: This needs custom accumulation logic for MLX
        raise NotImplementedError("Segment reduction not implemented for MLX")

    return result


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, mx.add, num_segments, sorted
    )


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    return _segment_reduction_fn(
        data, segment_ids, mx.maximum, num_segments, sorted
    )


def top_k(x, k, sorted=False):
    if sorted:
        sorted_indices = mx.argsort(x, axis=-1)[..., ::-1]
        sorted_values = mx.take_along_axis(x, sorted_indices, axis=-1)
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        # Note: MLX doesn't have partition, so we use full sort
        sorted_indices = mx.argsort(x, axis=-1)[..., ::-1]
        top_k_indices = sorted_indices[..., :k]
        top_k_values = mx.take_along_axis(x, top_k_indices, axis=-1)
    return top_k_values, top_k_indices


def in_top_k(targets, predictions, k):
    targets = mx.expand_dims(targets, axis=1)
    topk_values = top_k(predictions, k)[0]
    targets_values = mx.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return mx.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    max_x = mx.max(x, axis=axis, keepdims=True)
    exp_x = mx.exp(x - max_x)
    sum_exp_x = mx.sum(exp_x, axis=axis, keepdims=True)
    return mx.squeeze(max_x + mx.log(sum_exp_x), axis=axis) if not keepdims else max_x + mx.log(sum_exp_x)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return mx.linalg.qr(x)


def extract_sequences(x, sequence_length, sequence_stride):
    # Note: This needs custom implementation for MLX
    raise NotImplementedError("extract_sequences not implemented for MLX")


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    real, imag = x
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    dtype = standardize_dtype(real.dtype)
    if not "float" in dtype:
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    return real + 1j * imag


def _bit_reverse(n, bits):
    """Helper function for FFT to perform bit reversal"""
    rev = 0
    for i in range(bits):
        rev = (rev << 1) | (n & 1)
        n = n >> 1
    return rev

def fft(x):
    """Fast Fourier Transform implementation using Cooley-Tukey algorithm"""
    x = convert_to_tensor(x)
    n = x.shape[-1]
    if n & (n - 1): 
        raise ValueError("Length must be a power of 2 for this FFT implementation")
    
    bits = n.bit_length() - 1
    # Bit reversal permutation
    indices = mx.array([_bit_reverse(i, bits) for i in range(n)])
    x = mx.take(x, indices, axis=-1)
    
    # Cooley-Tukey FFT
    for stage in range(1, bits + 1):
        m = 1 << stage
        half_m = m // 2
        
        # Twiddle factors
        k = mx.arange(half_m)
        theta = -2 * mx.pi * k / m
        real = mx.cos(theta)
        imag = mx.sin(theta)
        
        for j in range(0, n, m):
            # Get segments
            even = x[..., j:j+half_m]
            odd = x[..., j+half_m:j+m]
            
            # Complex multiplication with twiddle factors
            real_part = odd * real - odd * imag
            imag_part = odd * real + odd * imag
            
            x = mx.concatenate([
                even + real_part,
                even - real_part
            ], axis=-1)
    
    return x.real, x.imag

def rfft(x, fft_length=None):
    """Real FFT implementation"""
    if fft_length is None:
        fft_length = x.shape[-1]
    
    # Pad if necessary
    if x.shape[-1] < fft_length:
        pad_size = fft_length - x.shape[-1]
        x = mx.pad(x, ((0, pad_size),))
    
    # Perform FFT
    real, imag = fft(x)
    
    # Return only positive frequencies (half + 1 of full FFT)
    n_freqs = fft_length // 2 + 1
    return real[..., :n_freqs], imag[..., :n_freqs]

def fft2(x):
    """2D FFT implementation using 1D FFT"""
    # First FFT along rows
    real_rows, imag_rows = fft(x)
    x_complex = mx.complex(real_rows, imag_rows)
    
    # Then FFT along columns
    x_t = mx.transpose(x_complex)
    real_cols, imag_cols = fft(x_t)
    
    # Transpose back
    result = mx.transpose(mx.complex(real_cols, imag_cols))
    return mx.real(result), mx.imag(result)

def stft(x, sequence_length, sequence_stride, fft_length, window="hann", center=True):
    """Short-time Fourier transform implementation"""
    if window == "hann":
        # Create Hann window
        window = 0.5 * (1 - mx.cos(2 * mx.pi * mx.arange(sequence_length) / (sequence_length - 1)))
    elif window == "hamming":
        # Create Hamming window
        window = 0.54 - 0.46 * mx.cos(2 * mx.pi * mx.arange(sequence_length) / (sequence_length - 1))
    
    if center:
        pad_width = fft_length // 2
        x = mx.pad(x, ((pad_width, pad_width),))
    
    # Extract segments
    n_segments = (x.shape[-1] - sequence_length) // sequence_stride + 1
    segments = []
    
    for i in range(n_segments):
        start = i * sequence_stride
        segment = x[..., start:start+sequence_length]
        if window is not None:
            segment = segment * window
        segments.append(segment)
    
    segments = mx.stack(segments, axis=0)
    
    # Perform FFT on each segment
    real, imag = rfft(segments, fft_length)
    return real, imag


def ifft(x):
    """Inverse Fast Fourier Transform implementation"""
    real, imag = x
    x = mx.complex(real, imag)
    n = x.shape[-1]
    
    if n & (n - 1):
        raise ValueError("Length must be a power of 2 for this IFFT implementation")
    
    bits = n.bit_length() - 1
    # Bit reversal permutation
    indices = mx.array([_bit_reverse(i, bits) for i in range(n)])
    x = mx.take(x, indices, axis=-1)
    
    # Cooley-Tukey IFFT (similar to FFT but with conjugate twiddle factors)
    for stage in range(1, bits + 1):
        m = 1 << stage
        half_m = m // 2
        
        # Twiddle factors (note positive angle for IFFT)
        k = mx.arange(half_m)
        theta = 2 * mx.pi * k / m
        real = mx.cos(theta)
        imag = mx.sin(theta)
        
        for j in range(0, n, m):
            even = x[..., j:j+half_m]
            odd = x[..., j+half_m:j+m]
            
            real_part = odd * real - odd * imag
            imag_part = odd * real + odd * imag
            
            x = mx.concatenate([
                even + real_part,
                even - real_part
            ], axis=-1)
    
    # Scale by 1/n
    x = x / n
    return mx.real(x), mx.imag(x)

def ifft2(x):
    """2D Inverse FFT implementation using 1D IFFT"""
    # First IFFT along rows
    real_rows, imag_rows = ifft(x)
    x_complex = mx.complex(real_rows, imag_rows)
    
    # Then IFFT along columns
    x_t = mx.transpose(x_complex)
    real_cols, imag_cols = ifft(x_t)
    
    # Transpose back
    result = mx.transpose(mx.complex(real_cols, imag_cols))
    return mx.real(result), mx.imag(result)

def irfft(x, fft_length=None):
    """Inverse Real FFT implementation"""
    real, imag = x
    if fft_length is None:
        fft_length = (real.shape[-1] - 1) * 2
    
    # Reconstruct negative frequencies using conjugate symmetry
    n_freqs = fft_length // 2 + 1
    real_full = mx.concatenate([real, mx.flip(real[..., 1:-1], -1)], axis=-1)
    imag_full = mx.concatenate([imag, -mx.flip(imag[..., 1:-1], -1)], axis=-1)
    
    # Perform IFFT
    result_real, result_imag = ifft((real_full, imag_full))
    return result_real  # Result should be real-valued

def istft(x, sequence_length, sequence_stride, fft_length, length=None, window="hann", center=True):
    """Inverse Short-time Fourier transform implementation"""
    if window == "hann":
        window = 0.5 * (1 - mx.cos(2 * mx.pi * mx.arange(sequence_length) / (sequence_length - 1)))
    elif window == "hamming":
        window = 0.54 - 0.46 * mx.cos(2 * mx.pi * mx.arange(sequence_length) / (sequence_length - 1))
    
    # Inverse FFT of each frame
    frames = irfft(x, fft_length)
    
    # Truncate to sequence_length
    frames = frames[..., :sequence_length]
    
    # Apply window
    if window is not None:
        frames = frames * window
    
    # Overlap-add
    n_frames = frames.shape[0]
    output_length = (n_frames - 1) * sequence_stride + sequence_length
    output = mx.zeros(output_length)
    
    for i in range(n_frames):
        start = i * sequence_stride
        output = output.at[start:start+sequence_length].add(frames[i])
    
    # Remove padding if center
    if center:
        start = fft_length // 2
        end = -fft_length // 2 if length is None else start + length
        output = output[start:end]
    
    # Normalize for window overlap
    if window is not None:
        # Calculate normalization factor based on window overlap
        win_sum = mx.zeros(output_length)
        for i in range(n_frames):
            start = i * sequence_stride
            win_sum = win_sum.at[start:start+sequence_length].add(window)
        # Avoid division by zero
        win_sum = mx.where(win_sum > 1e-10, win_sum, 1.0)
        output = output / win_sum
    
    return output

def erf(x):
    """Error function approximation using polynomial expansion"""
    # Constants for polynomial approximation
    a = mx.array([0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429])
    p = 0.3275911
    
    # Save the sign of x
    sign = mx.sign(x)
    x = mx.abs(x)
    
    # Formula 7.1.26 from Abramowitz and Stegun
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a[4] * t + a[3]) * t) + a[2]) * t + a[1]) * t + a[0]) * t * mx.exp(-x * x)
    
    return sign * y

def erfinv(x):
    """Inverse error function approximation using rational approximation"""
    # Rational approximation coefficients
    c = mx.array([1.0, 0.47047, 0.1216, 0.0273])
    d = mx.array([1.5774, 0.7379, 0.1089])
    
    # Save the sign of x
    sign = mx.sign(x)
    x = mx.abs(x)
    
    # Approximation for |x| <= 0.7
    def approx1(x):
        x2 = x * x
        num = ((c[3] * x2 + c[2]) * x2 + c[1]) * x2 + c[0]
        den = ((d[2] * x2 + d[1]) * x2 + d[0]) * x2 + 1.0
        return x * num / den
    
    # Approximation for |x| > 0.7
    def approx2(x):
        z = mx.sqrt(-mx.log((1 - x) / 2))
        return z - mx.log(z) / z
    
    # Combine approximations
    result = mx.where(x <= 0.7, approx1(x), approx2(x))
    return sign * result

def rsqrt(x):
    return 1.0 / mx.sqrt(x)


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    return mx.linalg.solve(a, b)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
    return mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims).astype(
        dtype
    )


def logdet(x):
    # Note: Using determinant directly since MLX doesn't have slogdet
    return mx.log(mx.linalg.det(x))
