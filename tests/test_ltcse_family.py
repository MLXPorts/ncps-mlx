import mlx.core as mx

from ncps.neurons import (
    CTRNNSECell,
    NODESECell,
    CTGRUSECell,
    CTRNN,
    CTGRU,
    NODE,
)


def test_ctrnn_se_forward():
    cell = CTRNNSECell(units=4)
    inputs = mx.random.normal((3, 5))
    output, state = cell(inputs)
    assert output.shape == (3, 4)
    assert state.shape == (3, 4)


def test_ctrnn_module_forward():
    module = CTRNN(input_size=5, units=4)
    inputs = mx.random.normal((2, 7, 5))
    outputs, state = module(inputs)
    assert outputs.shape == (2, 7, 4)
    assert state.shape == (2, 4)


def test_node_se_forward():
    cell = NODESECell(units=3)
    inputs = mx.random.normal((2, 6))
    output, state = cell(inputs)
    assert output.shape == (2, 3)
    assert state.shape == (2, 3)


def test_node_module_forward():
    module = NODE(input_size=6, units=3)
    inputs = mx.random.normal((1, 8, 6))
    outputs, state = module(inputs)
    assert outputs.shape == (1, 8, 3)
    assert state.shape == (1, 3)


def test_ctgru_se_forward():
    cell = CTGRUSECell(units=5)
    inputs = mx.random.normal((4, 7))
    output, state = cell(inputs)
    assert output.shape == (4, 5)
    assert state.shape == (4, 5 * cell.M)


def test_ctgru_module_forward():
    module = CTGRU(input_size=7, units=5)
    inputs = mx.random.normal((3, 9, 7))
    outputs, state = module(inputs)
    assert outputs.shape == (3, 9, 5)
    assert state.shape == (3, 5 * module.cell.M)
