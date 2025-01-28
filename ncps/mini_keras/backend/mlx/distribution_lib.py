"""Utilities for distribution strategy with Apple MLX backend."""
import mlx
try:
    import mlx.core.distributed as dist # type: ignore
except ImportError:
    from ncps.mini_keras.utils import logger
    logger.error("Apple MPI is not installed. Please install Apple MPI to use the distributed MLX backend.")
    raise


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Args:
        device_type: string of `"cpu"`, `"gpu"`, `"mlx"`. Defaults to `"gpu"`, `"mlx"`,
            or `"cpu"` if none is provided.

    Returns:
        List of devices that are available for distributed computation.
    """
    available_devices = mlx.devices()
    if device_type:
        filtered_devices = [dev for dev in available_devices if dev.type.lower() == device_type.lower()]
        return filtered_devices
    return available_devices


def initialize(job_addresses, num_processes, process_id):
    """Initialize the distribution system for multi-host/process setting.

    Args:
        job_addresses: string. Comma-separated IP addresses for all jobs in the cluster.
        num_processes: int. The total number of worker processes.
        process_id: int. The ID of the current worker/process.
    """
    dist.init(
        init_method=f"tcp://{job_addresses}",
        world_size=num_processes,
        rank=process_id,
    )


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: `np.ndarray` that needs to be distributed.
        layout: `TensorLayout` for the distribution information.

    Returns:
        Distributed value.
    """
    return dist.all_gather(tensor, layout)


def distribute_variable(value, layout):
    """Create a distributed variable for MLX.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable.

    Returns:
        Distributed variable.
    """
    return dist.DistributedVariable(value, layout)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Args:
        per_process_batch: `np.ndarray` already sharded to a local process size.
        layout: `TensorLayout` for the distribution information.
        batch_dim_name: Name of the batch dimension.

    Returns:
        A global batch distributed according to `layout`.
    """
    return dist.all_gather(per_process_batch, layout)


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return dist.get_world_size()


def process_id():
    """Return the current process ID for the distribution setting."""
    return dist.get_rank()


def _to_mlx_device(device_name):
    """Convert device name to MLX device.

    Args:
        device_name: Name of the device.

    Returns:
        MLX device.
    """
    if device_name.startswith("mps"):
        return mlx.Device("GPU")
    elif device_name.startswith("cpu"):
        return mlx.Device("CPU")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def _to_mlx_mesh(device_mesh):
    """Convert the DeviceMesh to MLX backend-specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A MLX Mesh instance.
    """
    return mlx.Mesh(device_mesh.devices, device_mesh.topology)


def _to_mlx_layout(tensor_layout):
    """Convert the TensorLayout to MLX backend-specific layout.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A MLX layout instance.
    """
    return mlx.Layout(tensor_layout.sharding_spec, tensor_layout.device_mesh)