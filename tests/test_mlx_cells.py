"""Regression tests for MLX liquid cells and layers."""

import mlx.core as mx

from ncps import CfC, CfCCell, CTRNNCell, CTGRUCell, LTCCell, LTC
from ncps.wirings import FullyConnected


def _build_wiring(input_dim: int, units: int, output_dim: int | None = None) -> FullyConnected:
    wiring = FullyConnected(units=units, output_dim=output_dim or units)
    wiring.build(input_dim)
    return wiring


def _clone_params(module):
    params = module.parameters()

    def clone(tree):
        if isinstance(tree, dict):
            return {key: clone(val) for key, val in tree.items()}
        if isinstance(tree, list):
            return [clone(val) for val in tree]
        if isinstance(tree, mx.array):
            return mx.array(tree)
        return tree

    return clone(params)


def _mx_allclose(lhs: mx.array, rhs: mx.array, tol: float = 1e-4) -> bool:
    return bool(mx.max(mx.abs(lhs - rhs)).item() < tol)


def test_cfc_cell_parameter_transfer():
    cell = CfCCell(input_size=4, hidden_size=6)
    x = mx.random.normal((2, 4))
    state = mx.zeros((2, 6))
    ts = mx.ones((2,))
    out, new_state = cell(x, state, ts)

    mx.eval(cell.parameters())
    snapshot = _clone_params(cell)

    clone = CfCCell(input_size=4, hidden_size=6)
    clone(x, state, ts)  # initialise parameters
    clone.update(snapshot)
    mx.eval(clone.parameters())
    out2, new_state2 = clone(x, state, ts)

    assert _mx_allclose(out, out2)
    assert _mx_allclose(new_state, new_state2)


def test_ltc_cell_parameter_transfer():
    wiring = _build_wiring(input_dim=3, units=5)
    cell = LTCCell(wiring=wiring, in_features=3)
    x = mx.random.normal((2, 3))
    state = mx.zeros((2, wiring.units))
    out, new_state = cell(x, state, mx.ones((2,)))

    mx.eval(cell.parameters())
    snapshot = _clone_params(cell)

    wiring_cfg = wiring.get_config()
    clone_wiring = type(wiring).from_config(wiring_cfg)
    clone_wiring.build(3)
    clone = LTCCell(wiring=clone_wiring, in_features=3)
    clone(x, state, mx.ones((2,)))
    clone.update(snapshot)
    mx.eval(clone.parameters())
    out2, new_state2 = clone(x, state, mx.ones((2,)))

    assert _mx_allclose(out, out2)
    assert _mx_allclose(new_state, new_state2)


def test_ctgru_cell_parameter_transfer():
    cell = CTGRUCell(units=4)
    x = mx.random.normal((2, 3))
    state = mx.zeros((2, 4 * cell.M))
    out, new_state = cell(x, state)

    mx.eval(cell.parameters())
    snapshot = _clone_params(cell)

    clone = CTGRUCell(units=4)
    clone(x, state)
    clone.update(snapshot)
    mx.eval(clone.parameters())
    out2, new_state2 = clone(x, state)

    assert _mx_allclose(out, out2)
    assert _mx_allclose(new_state, new_state2)


def test_ctrnn_cell_parameter_transfer():
    cell = CTRNNCell(units=4)
    x = mx.random.normal((2, 3))
    state = mx.zeros((2, 4))
    out, new_state = cell(x, state)

    mx.eval(cell.parameters())
    snapshot = _clone_params(cell)

    clone = CTRNNCell(units=4)
    clone(x, state)
    clone.update(snapshot)
    mx.eval(clone.parameters())
    out2, new_state2 = clone(x, state)

    assert _mx_allclose(out, out2)
    assert _mx_allclose(new_state, new_state2)


def test_cfc_layer_parameter_transfer():
    wiring = _build_wiring(input_dim=4, units=6)
    layer = CfC(input_size=4, units=wiring, return_sequences=True)
    data = mx.random.normal((2, 5, 4))
    layer.eval()
    output, state = layer(data)

    mx.eval(layer.parameters())
    snapshot = _clone_params(layer)
    wiring_cfg = wiring.get_config()
    restored_wiring = type(wiring).from_config(wiring_cfg)
    restored_wiring.build(4)
    restored = CfC(input_size=4, units=restored_wiring, return_sequences=True)
    restored(data)
    restored.update(snapshot)
    mx.eval(restored.parameters())
    restored.eval()
    output2, state2 = restored(data)

    assert _mx_allclose(output, output2)
    assert _mx_allclose(state, state2)


def test_ltc_layer_parameter_transfer():
    wiring = _build_wiring(input_dim=3, units=5)
    layer = LTC(input_size=3, units=wiring, return_sequences=True)
    data = mx.random.normal((2, 7, 3))
    layer.eval()
    output, state = layer(data)

    mx.eval(layer.parameters())
    snapshot = _clone_params(layer)
    wiring_cfg = wiring.get_config()
    restored_wiring = type(wiring).from_config(wiring_cfg)
    restored_wiring.build(3)
    restored = LTC(input_size=3, units=restored_wiring, return_sequences=True)
    restored(data)
    restored.update(snapshot)
    mx.eval(restored.parameters())
    restored.eval()
    output2, state2 = restored(data)

    assert _mx_allclose(output, output2)
    assert _mx_allclose(state, state2)
