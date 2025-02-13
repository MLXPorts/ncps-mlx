"""Tests for PaddlePaddle implementation."""

import paddle
import numpy as np
from ncps.paddle import LTCCell
from ncps.wirings import NCP


def test_ltc_cell():
    """Test LTCCell implementation."""
    # Create wiring
    wiring = NCP(
        inter_neurons=8,
        command_neurons=8,
        motor_neurons=4,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=4,
        motor_fanin=6
    )  # 8 neurons, 4 outputs
    
    # Create cell
    cell = LTCCell(
        wiring=wiring,
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        epsilon=1e-8,
        backbone_units=[16],
        backbone_layers=1
    )
    
    # Create test input
    batch_size = 2
    input_size = 8
    x = paddle.randn((batch_size, input_size))
    state = paddle.zeros((batch_size, wiring.units))
    
    # Build the cell
    print("\nBuilding cell...")
    cell.build((batch_size, input_size))
    print("Cell built successfully")
    
    # Test forward pass
    print("\nTesting forward pass...")
    output, new_state = cell(x, state)
    
    # Check shapes
    assert list(output.shape) == [batch_size, wiring.output_dim], \
        f"Expected output shape {[batch_size, wiring.output_dim]}, got {list(output.shape)}"
    assert list(new_state.shape) == [batch_size, wiring.units], \
        f"Expected state shape {[batch_size, wiring.units]}, got {list(new_state.shape)}"
    
    print("LTCCell tests passed!")


if __name__ == "__main__":
    test_ltc_cell()