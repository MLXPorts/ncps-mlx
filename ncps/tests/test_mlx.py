# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mlx.core as mx
import mlx.nn as nn
from ncps.mlx import CfC, LTCCell, LTC
from ncps import wirings
import numpy as np



def generate_data(N):
    """
    Generate synthetic time-series data.
    :param N: Length of the time-series.
    :return: Input and target tensors.
    """
    data_x = mx.array(
        np.stack(
            [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))],
            axis=1,
        ),
        dtype=mx.float32,
    )[None, :, :]  # Add batch dimension
    data_y = mx.array(
        np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]), dtype=mx.float32
    )
    return data_x, data_y


def test_fc():
    N = 48
    data_x, data_y = generate_data(N)

    fc_wiring = wirings.FullyConnected(8, 1)
    # Use LTC directly with solver specification
    ltc = LTC(
        fc_wiring,
        return_sequences=True,
        solver="rk4",  # Specify solver type
        ode_unfolds=6,
    )
    
    model = nn.Sequential([
        nn.InputLayer(input_shape=(None, 2)),
        ltc,
    ])
    
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_random():
    N = 48
    data_x, data_y = generate_data(N)

    arch = wirings.Random(32, 1, sparsity_level=0.5)
    ltc = LTC(arch, return_sequences=True)
    
    model = nn.Sequential([
        nn.InputLayer(input_shape=(None, 2)),
        ltc,
    ])
    
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_ncp():
    N = 48
    data_x, data_y = generate_data(N)

    ncp_wiring = wirings.NCP(
        inter_neurons=20,
        command_neurons=10,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=5,
        recurrent_command_synapses=6,
        motor_fanin=4,
    )
    ltc_cell = LTCCell(ncp_wiring)

    rnn = nn.RNN(cell=ltc_cell, return_sequences=True)
    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        rnn,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_fit_cfc():
    N = 48
    data_x, data_y = generate_data(N)

    rnn = CfC(8, return_sequences=True)
    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        rnn,
        nn.Linear(1),
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_mm_rnn():
    N = 48
    data_x, data_y = generate_data(N)

    rnn = CfC(8, return_sequences=True, mixed_memory=True)
    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        rnn,
        nn.Linear(1),
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_random_cfc():
    N = 48
    data_x, data_y = generate_data(N)

    arch = wirings.Random(32, 1, sparsity_level=0.5)
    cfc = CfC(arch, return_sequences=True)

    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        cfc,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_ncp_cfc_rnn():
    N = 48
    data_x, data_y = generate_data(N)

    ncp_wiring = wirings.NCP(
        inter_neurons=20,
        command_neurons=10,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=5,
        recurrent_command_synapses=6,
        motor_fanin=4,
    )
    cfc = CfC(ncp_wiring, return_sequences=True)

    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        cfc,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_auto_ncp_cfc_rnn():
    N = 48
    data_x, data_y = generate_data(N)

    ncp_wiring = wirings.AutoNCP(32, 1)
    cfc = CfC(ncp_wiring, return_sequences=True)

    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        cfc,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_ltc_rnn():
    N = 48
    data_x, data_y = generate_data(N)

    ltc = LTC(32, return_sequences=True)

    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        ltc,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)


def test_ncps():
    """Test basic NCPS functionality with LTCCell"""
    input_size = 8
    wiring = wirings.FullyConnected(8, 4)
    ltc_cell = LTCCell(wiring)
    
    data = mx.random.normal((3, input_size))
    hx = mx.zeros((3, wiring.units))
    output, hx = ltc_cell(data, hx)
    
    assert output.shape == (3, 4)
    assert hx.shape == (3, wiring.units)


def test_ncp_sizes():
    """Test NCP architecture sizing"""
    wiring = wirings.NCP(10, 10, 8, 6, 6, 4, 6)
    rnn = LTC(wiring)
    data = mx.random.normal((5, 3, 8))
    output = rnn(data)
    
    assert wiring.synapse_count > 0
    assert wiring.sensory_synapse_count > 0
    assert output.shape == (5, 8)


def test_auto_ncp():
    """Test AutoNCP functionality"""
    wiring = wirings.AutoNCP(16, 4)
    rnn = LTC(wiring)
    data = mx.random.normal((5, 3, 8))
    output = rnn(data)
    assert output.shape == (5, 4)


def test_enhanced_ltc():
    """Test enhanced LTC with different solvers"""
    wiring = wirings.FullyConnected(8, 4)
    ltc = LTC(wiring, solver="rk4", ode_unfolds=6)
    data = mx.random.normal((5, 3, 8))
    output = ltc(data)
    assert output.shape == (5, 4)


def test_mixed_memory_enhanced():
    """Test mixed memory with enhanced LTC"""
    N = 48
    data_x, data_y = generate_data(N)
    
    ltc = LTC(32, mixed_memory=True, solver="rk4", return_sequences=True)
    model = nn.Sequential(
        nn.InputLayer(input_shape=(None, 2)),
        ltc,
    )
    optimizer = nn.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=nn.MeanSquaredError())
    model.fit(data_x, data_y, batch_size=1, epochs=3)