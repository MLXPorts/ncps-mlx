import mlx.core as mx
import mlx.nn as nn

from ncps.neurons import (
    LTCCell,
    LTC,
    CfCCell,
    CfC,
    CTGRU,
    CTRNN,
    ELTCCell,
    ELTC,
    LTC4Cell,
    LTC4,
)
from ncps.wirings import AutoNCP, FullyConnected


def test_ltc_cell_forward():
    wiring = FullyConnected(units=6, output_dim=3)
    cell = LTCCell(wiring=wiring, in_features=4)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 4))
    state = mx.zeros((2, wiring.units), dtype=mx.float32)
    out, new_state = cell(x, state)
    assert out.shape == (2, wiring.output_dim)
    assert new_state.shape == (2, wiring.units)


def test_ltc_value_and_grad():
    wiring = FullyConnected(units=5, output_dim=5)
    model = LTC(input_size=3, units=wiring, return_sequences=True)

    def loss_fn(mdl, inputs, targets):
        preds, _ = mdl(inputs)
        return mx.mean((preds - targets) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(4, 7, 3))
    y = mx.random.uniform(low=-1.0, high=1.0, shape=(4, 7, wiring.output_dim))
    loss, grads = value_and_grad(model, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None


def test_cfc_cell_forward():
    cell = CfCCell(input_size=4, hidden_size=6)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(3, 4))
    state = mx.zeros((3, 6), dtype=mx.float32)
    out, new_state = cell(x, state, 1.0)
    assert out.shape == (3, 6)
    assert new_state.shape == (3, 6)


def test_cfc_wired_forward():
    wiring = AutoNCP(units=12, output_size=3, sparsity_level=0.5)
    rnn = CfC(input_size=5, units=wiring, return_sequences=True)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 9, 5))
    out, state = rnn(x)
    assert out.shape == (2, 9, wiring.output_dim)
    assert state.shape == (2, wiring.units)


def test_cfc_value_and_grad():
    rnn = CfC(input_size=4, units=8, return_sequences=False)

    def loss_fn(mdl, inputs, targets):
        preds, _ = mdl(inputs)
        return mx.mean((preds - targets) ** 2)

    value_and_grad = nn.value_and_grad(rnn, loss_fn)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(3, 10, 4))
    y = mx.random.uniform(low=-1.0, high=1.0, shape=(3, rnn.output_size))
    loss, grads = value_and_grad(rnn, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None


def test_ctgru_forward():
    rnn = CTGRU(input_size=3, units=6, return_sequences=True)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 7, 3))
    outputs, state = rnn(x)
    assert outputs.shape == (2, 7, 6)
    assert state.shape == (2, 6 * rnn.cell.M)


def test_ctgru_value_and_grad():
    rnn = CTGRU(input_size=2, units=4, return_sequences=False)

    def loss_fn(mdl, inputs, targets):
        preds, state = mdl(inputs)
        return mx.mean((preds - targets) ** 2) + 0.1 * mx.mean(state**2)

    value_and_grad = nn.value_and_grad(rnn, loss_fn)
    x = mx.random.uniform(low=-0.5, high=0.5, shape=(3, 6, 2))
    y = mx.random.uniform(low=-0.5, high=0.5, shape=(3, 4))
    loss, grads = value_and_grad(rnn, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None


def test_ctrnn_value_and_grad():
    rnn = CTRNN(input_size=4, units=5, return_sequences=False)

    def loss_fn(mdl, inputs, targets):
        preds, state = mdl(inputs)
        return mx.mean((preds - targets) ** 2) + mx.mean(state**2)

    value_and_grad = nn.value_and_grad(rnn, loss_fn)
    x = mx.random.uniform(low=-0.5, high=0.5, shape=(3, 9, 4))
    y = mx.random.uniform(low=-0.5, high=0.5, shape=(3, 5))
    loss, grads = value_and_grad(rnn, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None


def test_eltc_cell_forward():
    wiring = FullyConnected(units=8, output_dim=4)
    wiring.build(5)
    cell = ELTCCell(wiring=wiring, in_features=5, solver="explicit", ode_unfolds=3)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(4, 5))
    state = mx.zeros((4, wiring.units), dtype=mx.float32)
    out, new_state = cell(x, state, time=1.0)
    assert out.shape == (4, wiring.output_dim)
    assert new_state.shape == (4, wiring.units)


def test_eltc_value_and_grad():
    wiring = FullyConnected(units=6, output_dim=3)
    input_dim = 4
    rnn = ELTC(input_size=input_dim, wiring=wiring, return_sequences=True, solver="semi_implicit")

    def loss_fn(mdl, inputs, targets):
        preds = mdl(inputs)
        return mx.mean((preds - targets) ** 2)

    value_and_grad = nn.value_and_grad(rnn, loss_fn)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 5, input_dim))
    y = mx.random.uniform(low=-1.0, high=1.0, shape=(2, 5, wiring.output_dim))
    loss, grads = value_and_grad(rnn, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None


def test_ltc4_forward():
    cell = LTC4Cell(units=8, solver="rk4")
    x = mx.random.uniform(low=-1.0, high=1.0, shape=(5, 4))
    out, new_state = cell(x)
    assert out.shape == (5, 8)
    assert new_state.shape == (5, 8)


def test_ltc4_value_and_grad():
    model = LTC4(input_size=1, units=16, return_sequences=False)

    def loss_fn(mdl, inputs, targets):
        preds = mdl(inputs)
        return mx.mean((preds - targets) ** 2)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    x = mx.random.uniform(low=-0.5, high=0.5, shape=(6, 10, 1))
    y = mx.random.uniform(low=-0.5, high=0.5, shape=(6, 16))
    loss, grads = value_and_grad(model, x, y)
    assert isinstance(loss, mx.array)
    assert grads is not None
