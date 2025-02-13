"""Device-specific configurations for Neural Circuit Policy testing."""

from typing import Dict, List, Optional, Union
import platform
import subprocess

class DeviceConfig:
    """Base configuration class for Apple Silicon devices."""
    
    def __init__(
        self,
        device_type: str,
        batch_sizes: List[int],
        hidden_sizes: List[int],
        backbone_units: List[List[int]],
        memory_budget: int,
        min_tflops: float,
        min_bandwidth: float
    ):
        self.device_type = device_type
        self.batch_sizes = batch_sizes
        self.hidden_sizes = hidden_sizes
        self.backbone_units = backbone_units
        self.memory_budget = memory_budget  # MB
        self.min_tflops = min_tflops
        self.min_bandwidth = min_bandwidth  # GB/s

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for device."""
        return self.batch_sizes[-1]

    def get_optimal_hidden_size(self) -> int:
        """Get optimal hidden size for device."""
        return self.hidden_sizes[-1]

    def get_optimal_backbone(self) -> List[int]:
        """Get optimal backbone configuration for device."""
        return self.backbone_units[-1]

    def validate_performance(
        self,
        tflops: float,
        bandwidth: float,
        memory_usage: float
    ) -> bool:
        """Validate performance metrics meet device requirements."""
        return (
            tflops >= self.min_tflops and
            bandwidth >= self.min_bandwidth and
            memory_usage <= self.memory_budget
        )

class M1Config(DeviceConfig):
    """Configuration for M1 processor."""
    
    def __init__(self):
        super().__init__(
            device_type="M1",
            batch_sizes=[32, 64],
            hidden_sizes=[64, 128],
            backbone_units=[[32, 32], [64, 64]],
            memory_budget=8 * 1024,  # 8GB
            min_tflops=2.0,
            min_bandwidth=50.0
        )

class M1ProConfig(DeviceConfig):
    """Configuration for M1 Pro processor."""
    
    def __init__(self):
        super().__init__(
            device_type="M1 Pro",
            batch_sizes=[64, 128],
            hidden_sizes=[128, 256],
            backbone_units=[[64, 64], [128, 128]],
            memory_budget=16 * 1024,  # 16GB
            min_tflops=4.0,
            min_bandwidth=100.0
        )

class M1MaxConfig(DeviceConfig):
    """Configuration for M1 Max processor."""
    
    def __init__(self):
        super().__init__(
            device_type="M1 Max",
            batch_sizes=[128, 256],
            hidden_sizes=[256, 512],
            backbone_units=[[128, 128], [256, 256]],
            memory_budget=32 * 1024,  # 32GB
            min_tflops=8.0,
            min_bandwidth=200.0
        )

class M1UltraConfig(DeviceConfig):
    """Configuration for M1 Ultra processor."""
    
    def __init__(self):
        super().__init__(
            device_type="M1 Ultra",
            batch_sizes=[256, 512],
            hidden_sizes=[512, 1024],
            backbone_units=[[256, 256], [512, 512]],
            memory_budget=64 * 1024,  # 64GB
            min_tflops=16.0,
            min_bandwidth=400.0
        )

def detect_device() -> DeviceConfig:
    """Detect Apple Silicon device and return appropriate configuration."""
    try:
        # Run system_profiler on macOS
        result = subprocess.run(
            ['system_profiler', 'SPHardwareDataType'],
            capture_output=True,
            text=True
        )
        output = result.stdout.lower()
        
        # Detect device type
        if 'ultra' in output:
            return M1UltraConfig()
        elif 'max' in output:
            return M1MaxConfig()
        elif 'pro' in output:
            return M1ProConfig()
        else:
            return M1Config()
    except:
        # Default to M1 if detection fails
        return M1Config()

# Default configurations for each device type
DEVICE_CONFIGS = {
    'M1': M1Config(),
    'M1 Pro': M1ProConfig(),
    'M1 Max': M1MaxConfig(),
    'M1 Ultra': M1UltraConfig()
}

def get_device_config(device_type: Optional[str] = None) -> DeviceConfig:
    """Get configuration for specified device type or detect automatically."""
    if device_type is None:
        return detect_device()
    return DEVICE_CONFIGS[device_type]

class TestConfig:
    """Configuration for test execution."""
    
    def __init__(
        self,
        device_config: DeviceConfig,
        num_runs: int = 100,
        warmup_runs: int = 10,
        test_sizes: Optional[List[int]] = None,
        test_batch_sizes: Optional[List[int]] = None
    ):
        self.device_config = device_config
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.test_sizes = test_sizes or device_config.hidden_sizes
        self.test_batch_sizes = test_batch_sizes or device_config.batch_sizes
    
    def get_test_configs(self) -> List[Dict]:
        """Generate test configurations."""
        configs = []
        for size in self.test_sizes:
            for batch_size in self.test_batch_sizes:
                configs.append({
                    'hidden_size': size,
                    'batch_size': batch_size,
                    'backbone_units': [size, size],
                    'expected_tflops': self.device_config.min_tflops,
                    'expected_bandwidth': self.device_config.min_bandwidth,
                    'memory_budget': self.device_config.memory_budget
                })
        return configs

def create_test_config(
    device_type: Optional[str] = None,
    **kwargs
) -> TestConfig:
    """Create test configuration for device."""
    device_config = get_device_config(device_type)
    return TestConfig(device_config, **kwargs)