"""Tests for MLX array operations."""

import pytest
import numpy as np
import mlx.core as mx
from ncps.mlx.ops import array_ops


def test_slice():
    """Test slice operation."""
    x = array_ops.ones((4, 5))
    y = array_ops.slice(x, [1, 2], [2, 2])
    assert isinstance(y, mx.array)
    assert y.shape == (2, 2)

