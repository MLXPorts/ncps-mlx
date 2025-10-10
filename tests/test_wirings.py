import mlx.core as mx

from ncps.wirings import FullyConnected, Random, NCP, AutoNCP


def test_fully_connected_wiring_build():
    wiring = FullyConnected(units=4, output_dim=2, erev_init_seed=1234)
    wiring.build(3)

    assert isinstance(wiring.adjacency_matrix, mx.array)
    assert isinstance(wiring.sensory_adjacency_matrix, mx.array)
    assert wiring.adjacency_matrix.shape == (4, 4)
    assert wiring.sensory_adjacency_matrix.shape == (3, 4)
    synapse_total = int(mx.sum(mx.abs(wiring.adjacency_matrix)).item())
    assert synapse_total == 16  # fully connected including self connections


def test_random_wiring_reproducibility():
    w1 = Random(units=5, output_dim=2, sparsity_level=0.2, random_seed=42)
    w2 = Random(units=5, output_dim=2, sparsity_level=0.2, random_seed=42)
    w1.build(3)
    w2.build(3)

    assert bool(mx.all(w1.adjacency_matrix == w2.adjacency_matrix).item())
    assert bool(mx.all(w1.sensory_adjacency_matrix == w2.sensory_adjacency_matrix).item())


def test_ncp_layer_structure():
    wiring = NCP(
        inter_neurons=4,
        command_neurons=3,
        motor_neurons=2,
        sensory_fanout=2,
        inter_fanout=2,
        recurrent_command_synapses=1,
        motor_fanin=2,
        seed=1010,
    )
    wiring.build(5)

    assert wiring.num_layers == 3
    assert len(wiring.get_neurons_of_layer(0)) == 4
    assert len(wiring.get_neurons_of_layer(1)) == 3
    assert len(wiring.get_neurons_of_layer(2)) == 2


def test_autoncp_uses_mlx_arrays():
    wiring = AutoNCP(units=10, output_size=3, sparsity_level=0.4, seed=7)
    wiring.build(4)

    assert isinstance(wiring.adjacency_matrix, mx.array)
    assert isinstance(wiring.sensory_adjacency_matrix, mx.array)
    config = wiring.get_config()
    assert config["units"] == 10
    assert config["output_size"] == 3
