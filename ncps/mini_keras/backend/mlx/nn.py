import mlx.core as mx

from ncps.mini_keras import backend
from ncps.mini_keras.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_mlx,
)
from ncps.mini_keras.backend.mlx.core import cast
from ncps.mini_keras.backend.mlx.core import convert_to_tensor
from ncps.mini_keras.backend.mlx.core import is_tensor
from ncps.mini_keras.utils.module_utils import scipy


def relu(x):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(0.0, x.dtype))


def relu6(x):
    x = convert_to_tensor(x)
    # mx.clip incorrectly promote bfloat16 to float32, so we replace it with
    # mx.minimum and mx.maximum here
    return mx.minimum(
        mx.maximum(x, mx.array(0.0, x.dtype)), mx.array(6.0, x.dtype)
    )


def sigmoid(x):
    x = convert_to_tensor(x)
    return mx.array(1.0, x.dtype) / (mx.array(1.0, x.dtype) + mx.exp(-x))


def tanh(x):
    return mx.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return x - mx.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return mx.logaddexp(x, mx.array(0.0, x.dtype))


def softsign(x):
    x = convert_to_tensor(x)
    return x / (mx.array(1.0, x.dtype) + mx.abs(x))


def soft_shrink(x, threshold=0.5):
    return mx.where(
        x > threshold,
        mx.array(x - threshold, dtype=x.dtype),
        mx.where(
            x < -threshold,
            mx.array(x + threshold, dtype=x.dtype),
            mx.array(0.0, dtype=x.dtype),
        ),
    )


def sparse_plus(x):
    return mx.where(
        x <= -1,
        mx.zeros_like(x, dtype=x.dtype),
        mx.where(x < 1, mx.array((1 / 4) * (x + 1) ** 2, dtype=x.dtype), x),
    )


def silu(x):
    x = convert_to_tensor(x)
    return x * sigmoid(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + mx.sqrt(x**2 + b)
    return y / 2


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return -softplus(-x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(negative_slope, x.dtype) * x)


def hard_sigmoid(x):
    # python numbers will be promoted to float64 by np, so it's necessary to
    # first convert the python numbers to np scalars
    x = x / mx.array(6.0, x.dtype) + mx.array(0.5, x.dtype)
    return mx.where(
        x <= 0.0,
        mx.array(0.0, x.dtype),
        mx.where(x >= 1.0, mx.array(1.0, x.dtype), x),
    )


def hard_silu(x):
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return mx.where(
        x >= mx.array(0.0, x.dtype), x, mx.array(alpha, x.dtype) * mx.expm1(x)
    )


def selu(
    x,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
):
    x = convert_to_tensor(x)
    return mx.array(scale, x.dtype) * elu(x, alpha)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    # followed by JAX's implementation
    if approximate:
        sqrt_2_over_pi = mx.sqrt(2 / mx.pi).astype(x.dtype)
        cdf = mx.array(0.5, x.dtype) * (
            mx.array(1.0, x.dtype)
            + mx.tanh(
                sqrt_2_over_pi
                * (x + mx.array(0.044715, x.dtype) * (x**3).astype(x.dtype))
            )
        )
        return x * cdf
    else:
        sqrt_2 = mx.sqrt(2).astype(x.dtype)
        return (
            x
            * (scipy.special.erf(x / sqrt_2) + 1).astype(x.dtype)
            / mx.array(2, x.dtype)
        )


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    alpha = mx.array(alpha, x.dtype)
    return mx.maximum(x, mx.array(0.0, dtype=x.dtype)) + alpha * mx.expm1(
        mx.minimum(x, mx.array(0.0, dtype=x.dtype)) / alpha
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = mx.split(x, 2, axis)
    return x1 * (1 / (1 + mx.exp(-x2)))


def hard_tanh(x):
    x = convert_to_tensor(x)
    min_val = mx.asarray(-1.0, x.dtype)
    max_val = mx.asarray(1.0, x.dtype)
    return mx.array(mx.clip(x, min_val, max_val), dtype=x.dtype)


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    threshold = mx.asarray(threshold, x.dtype)
    return mx.array(
        mx.where(mx.abs(x) > threshold, x, mx.array(0.0, dtype=x.dtype)),
        dtype=x.dtype,
    )


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return mx.where(x > threshold, x, mx.array(default_value, dtype=x.dtype))


def softmax(x, axis=None):
    exp_x = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    max_x = mx.max(x, axis=axis, keepdims=True)
    logsumexp = mx.log(mx.exp(x - max_x).sum(axis=axis, keepdims=True))
    return x - max_x - logsumexp


def sparsemax(logits, axis=-1):
    # Sort logits along the specified axis in descending order
    logits = convert_to_tensor(logits)
    logits_sorted = -1.0 * mx.sort(-1.0 * logits, axis=axis)
    logits_cumsum = mx.cumsum(logits_sorted, axis=axis)
    r = mx.arange(1, logits.shape[axis] + 1)
    r_shape = [1] * logits.ndim
    r_shape[axis] = -1  # Broadcast to match the target axis
    r = r.reshape(r_shape)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    # Find the threshold
    k = mx.sum(support, axis=axis, keepdims=True)
    logits_cumsum_safe = mx.where(support, logits_cumsum, 0.0)
    tau = (mx.sum(logits_cumsum_safe, axis=axis, keepdims=True) - 1) / k
    output = mx.maximum(logits - tau, 0.0)
    return output


def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _pool(
    inputs,
    initial_value,
    reduce_fn,
    pool_size,
    strides=None,
    padding="valid",
):
    """Helper function to define pooling functions without using lax.reduce_window.
    
    Args:
        inputs: input data of shape (batch, height, width, channels) or 
               (batch, channels, height, width)
        initial_value: the initial value for the reduction
        reduce_fn: function to use for reduction (mx.max or mx.add)
        pool_size: tuple of 2 or 4 integers for pooling window size
        strides: tuple of 2 or 4 integers for stride size
        padding: "valid" or "same"
    
    Returns:
        Pooled output tensor
    """
    if padding.lower() not in ("valid", "same"):
        raise ValueError(f"Invalid padding '{padding}', must be 'same' or 'valid'")
    
    # Handle 4D vs 2D pool_size
    if len(pool_size) == 4:
        pool_h, pool_w = pool_size[1:3]
    else:
        pool_h, pool_w = pool_size
        
    if strides is None:
        strides = pool_size
        
    # Handle 4D vs 2D strides
    if len(strides) == 4:
        stride_h, stride_w = strides[1:3]
    else:
        stride_h, stride_w = strides

    # Get input dimensions
    if len(inputs.shape) != 4:
        raise ValueError("Input must be a 4D tensor")
        
    batch_size, in_height, in_width, channels = inputs.shape
    
    # Calculate output dimensions
    if padding.lower() == "valid":
        out_height = (in_height - pool_h) // stride_h + 1
        out_width = (in_width - pool_w) // stride_w + 1
    else:  # "same" padding
        out_height = (in_height + stride_h - 1) // stride_h
        out_width = (in_width + stride_w - 1) // stride_w
        
        pad_h = max(0, (out_height - 1) * stride_h + pool_h - in_height)
        pad_w = max(0, (out_width - 1) * stride_w + pool_w - in_width)
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        inputs = mx.pad(
            inputs,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            constant_values=initial_value
        )
        
    # Initialize output with initial value
    output = mx.full((batch_size, out_height, out_width, channels), initial_value, dtype=inputs.dtype)
    
    # Perform pooling operation
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride_h
            h_end = h_start + pool_h
            w_start = j * stride_w
            w_end = w_start + pool_w
            
            window = inputs[:, h_start:h_end, w_start:w_end, :]
            if reduce_fn == mx.max:
                output = output.at[:, i, j, :].set(
                    mx.max(window, axis=(1, 2))
                )
            else:  # mx.add for average pooling
                output = output.at[:, i, j, :].set(
                    mx.mean(window, axis=(1, 2))
                )
    
    return output

def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Max pooling operation.
    
    Args:
        inputs: Input tensor
        pool_size: Size of the pooling window
        strides: Stride size. If None, defaults to pool_size
        padding: "valid" or "same"
        data_format: "channels_last" or "channels_first"
    
    Returns:
        Pooled output tensor
    """
    data_format = backend.standardize_data_format(data_format)
    
    # Convert inputs to channels_last if needed
    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 3, 1))
        
    # Perform pooling
    outputs = _pool(
        inputs,
        initial_value=-mx.inf,
        reduce_fn=mx.max,
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )
    
    # Convert back to channels_first if needed
    if data_format == "channels_first":
        outputs = mx.transpose(outputs, (0, 3, 1, 2))
        
    return outputs

def average_pool(
    inputs,
    pool_size,
    strides,
    padding,
    data_format=None,
):
    """Average pooling operation.
    
    Args:
        inputs: Input tensor
        pool_size: Size of the pooling window
        strides: Stride size
        padding: "valid" or "same"
        data_format: "channels_last" or "channels_first"
    
    Returns:
        Pooled output tensor
    """
    data_format = backend.standardize_data_format(data_format)
    
    # Convert inputs to channels_last if needed
    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 3, 1))
        
    # Perform pooling
    outputs = _pool(
        inputs,
        initial_value=0.0,
        reduce_fn=mx.add,  # We'll use mean inside _pool for avg pooling
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )
    
    # Convert back to channels_first if needed
    if data_format == "channels_first":
        outputs = mx.transpose(outputs, (0, 3, 1, 2))
        
    return outputs


def _convert_to_lax_conv_dimension_numbers(
    num_spatial_dims,
    data_format="channels_last",
    transpose=False,
):
    """Create dimension numbers for convolution."""
    num_dims = num_spatial_dims + 2

    if data_format == "channels_last":
        spatial_dims = tuple(range(1, num_dims - 1))
        inputs_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        inputs_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return ConvDimensionNumbers(
        lhs_spec=inputs_dn, rhs_spec=kernel_dn, out_spec=inputs_dn
    )

class ConvDimensionNumbers:
    """Describes batch, spatial, and feature dimensions of a convolution."""
    def __init__(self, lhs_spec, rhs_spec, out_spec):
        self.lhs_spec = lhs_spec
        self.rhs_spec = rhs_spec
        self.out_spec = out_spec

def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    
    # Handle stride and dilation inputs
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )

    # Get shapes and validate
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
        
    kernel_in_channels = kernel.shape[-2]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    feature_group_count = channels // kernel_in_channels

    # Use MLX's native convolution
    return mx.conv(
        inputs,
        kernel if is_tensor(kernel) else kernel.numpy(),
        stride=strides,
        padding=padding.upper(),
        dilation=dilation_rate,
        feature_group_count=feature_group_count
    )

def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    """Depthwise 2D convolution.
    
    Args:
        inputs: Input tensor of shape (batch, height, width, channels) if data_format
            is "channels_last", or (batch, channels, height, width) if data_format
            is "channels_first".
        kernel: Kernel tensor of shape (kernel_height, kernel_width, in_channels,
            channel_multiplier).
        strides: Integer or tuple/list of 2 integers, specifying the strides of the
            convolution along the height and width.
        padding: "valid" or "same".
        data_format: "channels_last" or "channels_first".
        dilation_rate: Integer or tuple/list of 2 integers, specifying the dilation
            rate to use for dilated convolution.

    Returns:
        Output tensor.
    """
    data_format = backend.standardize_data_format(data_format)
    
    if not isinstance(strides, (list, tuple)):
        strides = (strides, strides)
    if not isinstance(dilation_rate, (list, tuple)):
        dilation_rate = (dilation_rate, dilation_rate)

    # Convert inputs to channels_last format if needed
    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 3, 1))

    batch_size, in_height, in_width, in_channels = inputs.shape
    kernel_h, kernel_w, _, channel_multiplier = kernel.shape
    
    # Reshape kernel for depthwise conv: (h, w, in_ch, ch_mult) -> (h, w, in_ch * ch_mult)
    kernel_reshaped = mx.reshape(kernel, 
                               (kernel_h, kernel_w, in_channels * channel_multiplier))

    # Perform depthwise convolution using MLX's conv operation
    outputs = mx.conv(
        inputs,
        kernel_reshaped,
        stride=strides,
        padding=padding.upper(),
        dilation=dilation_rate,
        feature_group_count=in_channels
    )

    # Reshape output to match the expected shape
    out_height, out_width = outputs.shape[1:3]
    outputs = mx.reshape(
        outputs,
        (batch_size, out_height, out_width, in_channels, channel_multiplier)
    )
    outputs = mx.reshape(
        outputs,
        (batch_size, out_height, out_width, in_channels * channel_multiplier)
    )

    # Convert back to channels_first format if needed
    if data_format == "channels_first":
        outputs = mx.transpose(outputs, (0, 3, 1, 2))

    return outputs

def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    padding_values = compute_conv_transpose_padding_args_for_mlx(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )

    # Convert stride and dilation inputs
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )

    # Use MLX's native transpose convolution
    return mx.conv_transpose(
        inputs,
        kernel if is_tensor(kernel) else kernel.numpy(),
        stride=strides,
        padding=padding_values,
        dilation=dilation_rate,
    )


def one_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    input_shape = x.shape

    x = x.reshape(-1)
    if not num_classes:
        num_classes = mx.max(x) + 1

    batch_size = x.shape[0]
    categorical = mx.zeros((batch_size, num_classes), dtype=dtype)
    valid_indices = x >= 0
    categorical[mx.arange(batch_size)[valid_indices], x[valid_indices]] = 1

    # First, reshape the array with the extra dimension at the end
    output_shape = input_shape + (num_classes,)
    categorical = mx.reshape(categorical, output_shape)

    # Then, move this new dimension to the right place (according to axis)
    if axis != -1:
        categorical = mx.moveaxis(categorical, -1, axis)

    return categorical


def multi_hot(x, num_classes, axis=-1, dtype="float32", sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with numpy backend")
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = mx.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = mx.array(target)
    output = mx.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis, keepdims=True)
        output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = mx.log(output)
    return -mx.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = mx.array(target, dtype="int32")
    output = mx.array(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = mx.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis, keepdims=True)
        output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = mx.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -mx.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = mx.array(target)
    output = mx.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = sigmoid(output)

    output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * mx.log(output)
    bce += (1.0 - target) * mx.log(1.0 - output)
    return -bce


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with NumPy."
        )
    axes = tuple(axes) if isinstance(axes, list) else axes
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = mx.mean(x, axes, keepdims=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    variance = mx.mean(mx.square(x), axis=axes, keepdims=True) - mx.square(mean)

    if not keepdims:
        mean = mx.squeeze(mean, axes)
        variance = mx.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = mx.clip(mean, mx.finfo(mx.float16).min, mx.finfo(mx.float16).max)
        variance = mx.clip(
            variance, mx.finfo(mx.float16).min, mx.finfo(mx.float16).max
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mx.reshape(mean, shape)
    variance = mx.reshape(variance, shape)

    inv = 1.0 / mx.sqrt(variance + epsilon)
    if scale is not None:
        scale = mx.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = mx.reshape(offset, shape)
        res = res + offset

    return x * inv + res


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    # Ref: https://github.com/google-deepmind/optax
    # optax.ctc_loss_with_forward_probs
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, "int32")
    output_length = convert_to_tensor(output_length, "int32")
    batch_size, max_input_length, num_classes = output.shape
    batch_size, max_label_length = target.shape
    log_epsilon = -1e5

    # Ensure that the dtype promotion behavior matches that of `tf.nn.ctc_loss`
    dtype = backend.result_type(output.dtype, "float32")
    output = output.astype(dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = mx.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = mx.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return mx.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(target_length, max_label_length)
    output_paddings = _lengths_to_paddings(output_length, max_input_length)
    target_paddings = target_paddings.astype(output.dtype)
    output_paddings = output_paddings.astype(output.dtype)

    logprobs = log_softmax(output, axis=-1)
    label_lengths = max_label_length - mx.sum(target_paddings, axis=1).astype(
        mx.int32
    )

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1].
    repeat = (target[:, :-1] == target[:, 1:]).astype(mx.float32)
    repeat = mx.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = mx.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = mx.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = mx.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    # [B, N]
    logalpha_phi_init = (
        mx.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    logalpha_phi_init[:, 0] = 0.0
    logalpha_emit_init = (
        mx.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        # Update `phi[:, 1:]`` with adding `added_score` in log space.
        return mx.concatenate(
            [phi[:, :1], mx.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        # emit-to-phi epsilon transition, except if the next label is repetition
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(prev_phi, prev_emit + log_epsilon * repeat)

        logprob_emit, logprob_phi, pad = x

        # phi-to-emit transition
        next_emit = mx.logaddexp(
            prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )
        # self-loop transition
        next_phi = prev_phi + logprob_phi
        # emit-to-phi blank transition only when the next label is repetition
        next_phi = update_phi_score(
            next_phi, prev_emit + logprob_phi + log_epsilon * (1.0 - repeat)
        )

        pad = pad.reshape((batch_size, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    def np_scan(f, init, xs):
        carry = init
        ys = []
        for x in zip(*xs):
            carry, y = f(carry, x)
            ys.append(y)
        result = []
        for i in range(len(ys[0])):
            result.append(mx.stack([y[i] for y in ys]))
        return carry, result

    xs = (logprobs_emit, logprobs_phi, output_paddings.transpose((1, 0)))
    _, (logalpha_phi, logalpha_emit) = np_scan(
        loop_body, (logalpha_phi_init, logalpha_emit_init), xs
    )

    # last row needs to be updated with the last epsilon transition
    logalpha_phi_last = update_phi_score(logalpha_phi[-1], logalpha_emit[-1])
    logalpha_phi[-1] = logalpha_phi_last

    # extract per_seq_loss
    # [B, N+1]
    _one_hot = one_hot(label_lengths, num_classes=max_label_length + 1)
    per_seq_loss = -mx.einsum("bn,bn->b", logalpha_phi_last, _one_hot)
    return per_seq_loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = mx.argmax(inputs, axis=-1).astype("int32")
    scores = mx.max(inputs, axis=-1)

    seqlen_mask = mx.arange(max_length)[None, :]
    seqlen_mask = seqlen_mask >= sequence_lengths[:, None]

    indices = mx.where(seqlen_mask, mask_index, indices)
    scores = mx.where(seqlen_mask, 0.0, scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = mx.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = mx.where(repeat_mask, mask_index, indices)

    # We set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = mx.where(invalid_mask, -1, indices)

    # We rearrange the indices by moving `mask_index` to the end of the array
    order = mx.expand_dims(mx.arange(max_length), axis=0)  # [1, N]
    order = mx.tile(order, (batch_size, 1))  # [B, N]
    order = mx.where(invalid_mask, max_length, order)
    order = mx.argsort(order, axis=-1)
    indices = mx.take_along_axis(indices, order, axis=-1)

    scores = -mx.sum(scores, axis=1)[:, None]
    indices = mx.expand_dims(indices, axis=0)
    return indices, scores


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths)

    batch_size, max_seq_len, num_classes = inputs.shape
    inputs = log_softmax(inputs, axis=-1)
    seqlen_mask = mx.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]

    if mask_index is None:
        mask_index = num_classes - 1

    # This is a workaround for the fact that mx.argsort does not support
    # the order parameter which is used to break ties when scores are equal.
    # For compatibility with the tensorflow implementation, we flip the inputs
    # and the mask_index, and then flip the classes back to the correct indices
    inputs = mx.flip(inputs, axis=2)
    mask_index = num_classes - mask_index - 1

    _pad = -1

    init_paths = mx.full(
        (batch_size, 2 * beam_width, max_seq_len), _pad, dtype=mx.int32
    )

    num_init_paths = mx.min(mx.array([num_classes, beam_width]))
    max_classes = mx.argsort(inputs[:, 0], axis=1)[:, -num_init_paths:]
    init_classes = mx.where(max_classes == mask_index, _pad, max_classes)
    init_paths[:, :num_init_paths, 0] = init_classes

    init_scores = mx.full(
        (batch_size, 2 * beam_width), -mx.inf, dtype=inputs.dtype
    )
    init_scores[:, :num_init_paths] = mx.take_along_axis(
        inputs[:, 0], max_classes, axis=1
    )
    init_masked = init_paths[:, :, 0] == _pad

    def _extend_paths(paths, scores, masked, x):
        paths = mx.repeat(paths, num_classes, axis=0)
        scores = mx.repeat(scores, num_classes)
        masked = mx.repeat(masked, num_classes)

        path_tail_index = mx.argmax(paths == _pad, axis=1)
        paths_arange = mx.arange(2 * beam_width * num_classes)
        path_tails = paths[paths_arange, path_tail_index - 1]
        path_tails = mx.where(path_tail_index == 0, _pad, path_tails)

        classes = mx.arange(num_classes)
        classes[mask_index] = _pad
        classes = mx.tile(classes, 2 * beam_width)

        prev_masked = masked
        masked = classes == _pad

        masked_repeat = ~prev_masked & (path_tails == classes)
        classes = mx.where(masked_repeat, _pad, classes)
        paths[paths_arange, path_tail_index] = classes

        x = mx.tile(x, 2 * beam_width)
        scores = scores + x

        return paths, scores, masked

    def _merge_scores(unique_inverse, scores):
        scores_max = mx.max(scores)
        scores_exp = mx.exp(scores - scores_max)
        scores = mx.zeros_like(scores)
        for i, u in enumerate(unique_inverse):
            scores[u] += scores_exp[i]
        scores = mx.log(scores) + scores_max
        return scores

    def _prune_paths(paths, scores, masked):
        paths, unique_inverse = mx.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = mx.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = mx.squeeze(unique_inverse, axis=1)

        emit_scores = mx.where(masked, -mx.inf, scores)
        mask_scores = mx.where(masked, scores, -mx.inf)

        emit_scores = _merge_scores(unique_inverse, emit_scores)
        mask_scores = _merge_scores(unique_inverse, mask_scores)

        total_scores = mx.logaddexp(emit_scores, mask_scores)
        top_indices = mx.argsort(total_scores, kind="stable")[-beam_width:]

        paths = paths[top_indices]
        emit_scores = emit_scores[top_indices]
        mask_scores = mask_scores[top_indices]

        paths = mx.tile(paths, (2, 1))
        scores = mx.concatenate([emit_scores, mask_scores])
        masked = mx.concatenate(
            [mx.zeros(beam_width, bool), mx.ones(beam_width, bool)]
        )

        return paths, scores, masked

    def _decode_step(paths, scores, masked, x):
        paths, scores, masked = _extend_paths(paths, scores, masked, x)
        paths, scores, masked = _prune_paths(paths, scores, masked)
        return paths, scores, masked

    def _step(prev, x):
        paths, scores, masked = prev
        x, seqlen_mask = x
        if not seqlen_mask:
            paths, scores, masked = _decode_step(paths, scores, masked)
        return (paths, scores, masked), None

    def _decode_batch(
        init_paths, init_scores, init_masked, inputs, seqlen_mask
    ):
        def np_scan_only_carry(f, init, xs):
            carry = init
            for x in zip(*xs):
                carry, y = f(carry, x)
            return carry, None

        (paths, scores, masked), _ = np_scan_only_carry(
            _step,
            (init_paths, init_scores, init_masked),
            (inputs[1:], seqlen_mask[1:]),
        )

        paths, unique_inverse = mx.unique(paths, return_inverse=True, axis=0)
        pad_size = (2 * num_classes * beam_width) - len(paths)
        if pad_size > 0:
            paths = mx.pad(paths, [[0, pad_size], [0, 0]], constant_values=_pad)
        paths = paths[: 2 * num_classes * beam_width]
        if len(unique_inverse.shape) >= 2:
            unique_inverse = mx.squeeze(unique_inverse, axis=1)
        scores = _merge_scores(unique_inverse, scores)

        top_indices = mx.argsort(scores)[-top_paths:][::-1]
        paths = paths[top_indices]
        scores = scores[top_indices]

        return paths, scores

    results = [
        _decode_batch(p, s, m, i, sm)
        for p, s, m, i, sm in zip(
            init_paths, init_scores, init_masked, inputs, seqlen_mask
        )
    ]
    paths = mx.stack([r[0] for r in results])
    scores = mx.stack([r[1] for r in results])

    # convert classes back to the correct indices
    paths = mx.where(paths == _pad, _pad, num_classes - paths - 1)
    paths = mx.transpose(paths, [1, 0, 2])
    return paths, scores


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )


def psnr(x1, x2, max_val):
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = mx.mean(mx.square(x1 - x2))
    psnr = 20 * mx.log10(max_val) - 10 * mx.log10(mse)
    return psnr


def _get_large_negative(dtype):
    dtype = backend.standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return mx.asarray(val * -0.7, dtype=dtype)


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = mx.ones_like(logits, dtype=mx.bool_)
    if mask is not None:
        combined_mask = mx.logical_and(combined_mask, mask)

    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        mask = mx.tril(mx.ones((T, S), dtype=mx.bool_))
        mask = mask[None, None, :, :]
        combined_mask = mx.logical_and(combined_mask, mask)

    padded_logits = mx.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )
    return padded_logits


def _dot_product_attention_xla(query, key, value, bias, mask, is_causal, scale):
    original_dtype = key.dtype
    logits_dtype = mx.promote_types(query.dtype, mx.float32)
    if backend.standardize_dtype(key.dtype) == "bfloat16":
        # `mx.einsum` doesn't support bfloat16
        key = key.astype("float32")
        value = value.astype("float32")
    logits = mx.einsum("BTNH,BSNH->BNTS", query, key)
    logits = logits.astype(logits_dtype)
    logits *= mx.array(scale, dtype=logits.dtype)

    if bias is not None:
        logits = (logits + bias).astype(logits.dtype)

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax and it is always carried out in fp32.
    padded_logits = padded_logits.astype(mx.float32)
    probs = softmax(padded_logits, axis=-1).astype(original_dtype)
    encoded_dtype = probs.dtype
    if backend.standardize_dtype(probs.dtype) == "bfloat16":
        # `mx.einsum` doesn't support bfloat16
        probs = probs.astype("float32")
        value = value.astype("float32")
    encoded = mx.einsum("BNTS,BSNH->BTNH", probs, value)
    encoded = encoded.astype(encoded_dtype)
    return encoded


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
):
    if flash_attention is None:
        flash_attention = False
    if flash_attention:
        raise ValueError("Flash attention is not supported in numpy backend.")

    # Ref: jax.nn.dot_product_attention
    # https://github.com/jax-ml/jax/blob/jax-v0.4.32/jax/_src/nn/functions.py#L828
    # Not support `query_seq_lengths` and `key_value_seq_lengths` args
    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    _, _, _, H = key.shape
    scale = (1.0 / mx.sqrt(H)) if scale is None else scale
    return _dot_product_attention_xla(
        query, key, value, bias, mask, is_causal, scale
    )
