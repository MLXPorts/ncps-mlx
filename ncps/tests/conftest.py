"""Test configuration for NCPS."""

import pytest
import keras
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    # For Keras 3.8, we only need numpy seed since we're using numpy for random ops
    # The random ops in Keras will use numpy's random state


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 8


@pytest.fixture
def input_dim():
    """Input dimension for testing."""
    return 64


@pytest.fixture
def state_size():
    """State size for testing."""
    return 32


@pytest.fixture
def seq_len():
    """Sequence length for testing."""
    return 10


@pytest.fixture
def time_delta():
    """Time delta values for testing."""
    return 0.1


@pytest.fixture
def input_tensor(batch_size, input_dim):
    """Input tensor for testing."""
    return keras.ops.ones((batch_size, input_dim))


@pytest.fixture
def state_tensor(batch_size, state_size):
    """State tensor for testing."""
    return keras.ops.zeros((batch_size, state_size))


@pytest.fixture
def sequence_tensor(batch_size, seq_len, input_dim):
    """Sequence tensor for testing."""
    return keras.ops.ones((batch_size, seq_len, input_dim))


@pytest.fixture
def time_tensor(batch_size, seq_len):
    """Time tensor for testing."""
    return keras.ops.ones((batch_size, seq_len, 1))


class DummyWiring:
    """Dummy wiring configuration for testing."""
    
    def __init__(self, units, output_dim=None):
        self.units = units
        self.output_dim = output_dim or units
        
    def get_config(self):
        return {
            "units": self.units,
            "output_dim": self.output_dim
        }


@pytest.fixture
def wiring(state_size):
    """Wiring configuration for testing."""
    return DummyWiring(state_size)


def assert_close(x, y, rtol=1e-5, atol=1e-8):
    """Assert tensors are close."""
    x = keras.ops.convert_to_numpy(x)
    y = keras.ops.convert_to_numpy(y)
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def assert_shape_equal(x, expected_shape):
    """Assert tensor has expected shape."""
    assert x.shape == expected_shape, f"Expected shape {expected_shape}, got {x.shape}"


def assert_dtype_equal(x, expected_dtype):
    """Assert tensor has expected dtype."""
    assert x.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {x.dtype}"


@pytest.fixture
def assert_helpers():
    """Helper functions for assertions."""
    return {
        "assert_close": assert_close,
        "assert_shape_equal": assert_shape_equal,
        "assert_dtype_equal": assert_dtype_equal
    }


class TestConfig:
    """Test configuration values."""
    
    backbone_units = 128
    backbone_layers = 2
    backbone_dropout = 0.1
    solver_method = "rk4"
    activation = "tanh"


@pytest.fixture
def test_config():
    """Configuration for testing."""
    return TestConfig


def make_time_sequence(batch_size, seq_len, pattern="constant"):
    """Create time sequence for testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        pattern: Time pattern ("constant", "linear", "random")
    """
    if pattern == "constant":
        return np.ones((batch_size, seq_len, 1))
    elif pattern == "linear":
        t = np.linspace(0, 1, seq_len)
        return np.broadcast_to(t[None, :, None], (batch_size, seq_len, 1))
    elif pattern == "random":
        return np.random.uniform(0, 1, (batch_size, seq_len, 1))
    else:
        raise ValueError(f"Unknown time pattern: {pattern}")


@pytest.fixture
def make_time():
    """Function to create time sequences."""
    return make_time_sequence


def compare_models(model1, model2, input_shape, rtol=1e-5):
    """Compare outputs of two models.
    
    Args:
        model1: First model
        model2: Second model
        input_shape: Input shape for test
        rtol: Relative tolerance
    """
    # Create random input
    x = np.random.normal(size=input_shape)
    
    # Get outputs
    y1 = model1(x)
    y2 = model2(x)
    
    # Compare
    assert_close(y1, y2, rtol=rtol)


@pytest.fixture
def model_compare():
    """Function to compare models."""
    return compare_models