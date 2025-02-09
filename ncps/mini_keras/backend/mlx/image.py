import mlx.core as mx
from mlx.utils import tree_map

def convert_to_tensor(x):
    if not isinstance(x, mx.array):
        return mx.array(x)
    return x

RESIZE_INTERPOLATIONS = ("bilinear", "nearest", "lanczos3", "lanczos5", "bicubic")

def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = "channels_last" if data_format is None else data_format
    channels_axis = -1 if data_format == "channels_last" else -3
    
    if images.ndim not in (3, 4):
        raise ValueError(f"Invalid images rank: {images.shape}")

    original_dtype = images.dtype
    compute_dtype = mx.float32 if mx.issubdtype(original_dtype, mx.integer) else original_dtype
    images = images.astype(compute_dtype)

    rgb_weights = mx.array([0.2989, 0.5870, 0.1140], dtype=images.dtype)
    images = mx.tensordot(images, rgb_weights, axes=([channels_axis], [0]))
    return mx.expand_dims(images, axis=channels_axis).astype(original_dtype)

def rgb_to_hsv(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = "channels_last" if data_format is None else data_format
    channels_axis = -1 if data_format == "channels_last" else -3
    
    if images.ndim not in (3, 4):
        raise ValueError(f"Invalid images rank: {images.shape}")
    
    if not mx.issubdtype(images.dtype, mx.floating):
        raise ValueError("Input images must be float type")

    eps = mx.finfo(images.dtype).eps
    images = mx.where(mx.abs(images) < eps, 0.0, images)
    
    split_axis = -1 if data_format == "channels_last" else -3
    red, green, blue = mx.split(images, 3, axis=split_axis)
    red = mx.squeeze(red, split_axis)
    green = mx.squeeze(green, split_axis)
    blue = mx.squeeze(blue, split_axis)

    v = mx.maximum(mx.maximum(red, green), blue)
    m = mx.minimum(mx.minimum(red, green), blue)
    diff = v - m

    safe_v = mx.where(v > 0, v, 1.0)
    sat = mx.where(v > 0, diff / safe_v, 0.0)
    sat = mx.where(diff == 0, 0.0, sat)

    hr = mx.where(diff == 0, 0, (green - blue) / diff)
    hg = (blue - red) / diff + 2.0
    hb = (red - green) / diff + 4.0
    
    h = mx.where(v == red, hr, mx.where(v == green, hg, hb))
    h = (h / 6.0) % 1.0
    
    return mx.stack([h, sat, v], axis=channels_axis)

def hsv_to_rgb(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = "channels_last" if data_format is None else data_format
    channels_axis = -1 if data_format == "channels_last" else -3
    
    split_axis = -1 if data_format == "channels_last" else -3
    h, s, v = mx.split(images, 3, axis=split_axis)
    h = mx.squeeze(h, split_axis)
    s = mx.squeeze(s, split_axis)
    v = mx.squeeze(v, split_axis)

    h = (h % 1.0) * 6.0
    i = mx.floor(h)
    f = h - i
    i = i.astype(mx.int32)

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = mx.stack([
        mx.where(i == 5, v, mx.where(i == 4, q, mx.where(i == 3, p, 
        mx.where(i == 2, p, mx.where(i == 1, q, v))))),
        mx.where(i == 0, t, mx.where(i == 1, v, mx.where(i == 2, v, 
        mx.where(i == 3, q, mx.where(i == 4, p, p))))),
        mx.where(i == 0, p, mx.where(i == 1, p, mx.where(i == 2, t, 
        mx.where(i == 3, v, mx.where(i == 4, v, q)))))
    ], axis=channels_axis)

    return mx.where(s[..., None] == 0, v[..., None], rgb)

def resize(images, size, interpolation="bilinear", data_format=None):
    data_format = "channels_last" if data_format is None else data_format
    
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(f"Invalid interpolation: {interpolation}")
    
    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1) if images.ndim == 4 else (1, 2, 0))

    resized = mx.image.resize(images, size, interpolation=interpolation)
    
    if data_format == "channels_first":
        resized = mx.transpose(resized, (0, 3, 1, 2) if resized.ndim == 4 else (2, 0, 1))
    
    return resized

def affine_transform(images, transform, interpolation="bilinear", fill_value=0, data_format=None):
    data_format = "channels_last" if data_format is None else data_format
    original_shape = images.shape
    
    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1) if images.ndim == 4 else (1, 2, 0))

    batch_size = images.shape[0] if images.ndim == 4 else 1
    height, width = images.shape[1:3] if images.ndim ==4 else images.shape[:2]
    
    # Generate grid
    x = mx.linspace(-1, 1, width)
    y = mx.linspace(-1, 1, height)
    x_t, y_t = mx.meshgrid(x, y)
    ones = mx.ones_like(x_t)
    grid = mx.stack([x_t, y_t, ones], axis=-1)

    # Apply transformation
    transform = mx.reshape(transform, (-1, 3, 2))
    grid = mx.matmul(grid, transform)
    
    # Resample
    output = mx.zeros_like(images)
    for b in range(batch_size):
        output[b] = mx.nn.functional.grid_sample(
            images[b][None],
            grid[None],
            mode=interpolation,
            padding_mode='zeros'
        )[0]

    if data_format == "channels_first":
        output = mx.transpose(output, (0, 3, 1, 2) if output.ndim == 4 else (2, 0, 1))
    
    return output

def map_coordinates(inputs, coordinates, order=1, fill_value=0.0):
    if order != 1:
        raise NotImplementedError("Only linear interpolation (order=1) is supported")
        
    return mx.nn.functional.grid_sample(
        inputs[None], 
        coordinates[None], 
        mode='linear', 
        padding_value=fill_value
    )[0]
