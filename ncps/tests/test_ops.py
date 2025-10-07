"""Tests for NumPy-based operations."""

import pytest
import numpy as np
from ncps import ops


def test_array_ops():
    """Test array operations."""
    # Test convert_to_tensor
    x = [[1, 2], [3, 4]]
    t = ops.convert_to_tensor(x)
    assert isinstance(t, np.ndarray)
    assert t.shape == (2, 2)
    
    # Test reshape
    t = ops.reshape(t, (4,))
    assert t.shape == (4,)
    
    # Test concatenate
    x1 = np.ones((2, 3))
    x2 = np.zeros((2, 3))
    t = ops.concatenate([x1, x2], axis=0)
    assert t.shape == (4, 3)
    
    # Test stack
    t = ops.stack([x1, x2], axis=0)
    assert t.shape == (2, 2, 3)
    
    # Test split
    parts = ops.split(t, 2, axis=0)
    assert len(parts) == 2
    assert all(p.shape == (1, 2, 3) for p in parts)


def test_math_ops():
    """Test mathematical operations."""
    x = np.array([1., 2., 3.])
    y = np.array([4., 5., 6.])
    
    # Test basic arithmetic
    assert np.allclose(ops.add(x, y), [5., 7., 9.])
    assert np.allclose(ops.subtract(x, y), [-3., -3., -3.])
    assert np.allclose(ops.multiply(x, y), [4., 10., 18.])
    assert np.allclose(ops.divide(x, y), [0.25, 0.4, 0.5])
    
    # Test reductions
    assert np.isclose(ops.reduce_mean(x), 2.0)
    assert np.isclose(ops.reduce_sum(x), 6.0)
    assert np.isclose(ops.reduce_max(x), 3.0)
    assert np.isclose(ops.reduce_min(x), 1.0)
    
    # Test element-wise ops
    assert np.allclose(ops.abs([-1., 1.]), [1., 1.])
    assert np.allclose(ops.exp([0., 1.]), [1., np.e])
    assert np.allclose(ops.sqrt([1., 4.]), [1., 2.])


def test_nn_ops():
    """Test neural network operations."""
    x = np.array([-1., 0., 1.])
    
    # Test activations
    assert np.allclose(ops.sigmoid(x), 1 / (1 + np.exp(-x)))
    assert np.allclose(ops.tanh(x), np.tanh(x))
    assert np.allclose(ops.relu(x), [0., 0., 1.])
    
    # Test softmax
    x = np.array([[1., 2.], [3., 4.]])
    s = ops.softmax(x)
    assert np.allclose(np.sum(s, axis=-1), [1., 1.])
    
    # Test dropout
    x = np.ones((1000, 100))
    y = ops.dropout(x, rate=0.5, training=True)
    # Check approximate number of zeros
    zeros = np.sum(y == 0) / y.size
    assert 0.45 <= zeros <= 0.55


def test_random_ops():
    """Test random operations."""
    # Test seeding
    ops.set_seed(42)
    x1 = ops.normal((1000,))
    ops.set_seed(42)
    x2 = ops.normal((1000,))
    assert np.allclose(x1, x2)
    
    # Test distributions
    x = ops.normal((1000,))
    assert -0.1 < np.mean(x) < 0.1  # approximately zero mean
    assert 0.9 < np.std(x) < 1.1    # approximately unit variance
    
    x = ops.uniform((1000,))
    assert np.all(x >= 0) and np.all(x <= 1)
    
    x = ops.bernoulli((1000,), p=0.7)
    assert 0.65 < np.mean(x) < 0.75


def test_state_ops():
    """Test state operations."""
    # Test Variable creation
    v = ops.Variable(np.zeros((2, 3)))
    assert v.shape == (2, 3)
    assert np.allclose(v.value, 0)
    
    # Test assign
    ops.assign(v, np.ones((2, 3)))
    assert np.allclose(v.value, 1)
    
    # Test scatter update
    ops.scatter_update(v, [0], [[2, 2, 2]])
    assert np.allclose(v.value[0], 2)
    assert np.allclose(v.value[1], 1)
    
    # Test scatter add
    ops.scatter_add(v, [1], [[1, 1, 1]])
    assert np.allclose(v.value[0], 2)
    assert np.allclose(v.value[1], 2)


def test_complex_ops():
    """Test complex combinations of operations."""
    # Create random input
    x = ops.normal((10, 5))
    
    # Apply series of operations
    y = ops.relu(x)
    y = ops.dropout(y, rate=0.5, training=True)
    y = ops.layer_normalization(y)
    
    # Check shapes maintained
    assert y.shape == x.shape
    
    # Test backprop-style computation
    w = ops.Variable(ops.normal((5, 3)))
    b = ops.Variable(ops.zeros((3,)))
    
    def forward(x):
        h = ops.matmul(x, w.value)
        h = ops.add(h, b.value)
        h = ops.relu(h)
        return h
    
    # Run forward pass
    h = forward(x)
    assert h.shape == (10, 3)
    
    # Update weights (simulate gradient update)
    grad = ops.normal(w.shape)
    ops.assign_sub(w, 0.1 * grad)
    
    # Check update happened
    h_new = forward(x)
    assert not np.allclose(h, h_new)