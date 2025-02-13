import numpy as np
import keras
from ncps.keras import LTC, EnhancedLTCCell  # Changed to keras implementation
from ncps import wirings

def generate_data(N):
    in_features = 2
    out_features = 1
    # Input feature is a sine and a cosine wave
    time = np.linspace(0, 3 * np.pi, N)
    data_x = np.stack(
        [np.sin(time), np.cos(time)], axis=1
    )
    data_x = np.expand_dims(data_x, axis=0)  # Add batch dimension
    # Target output is a sine with double the frequency of the input signal
    data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1])
    return data_x.astype(np.float32), data_y.astype(np.float32)

N = 48
data_x, data_y = generate_data(N)

# Create NCP wiring configuration
ncp_wiring = wirings.NCP(
    inter_neurons=20,
    command_neurons=10,
    motor_neurons=1,
    sensory_fanout=4,
    inter_fanout=5,
    recurrent_command_synapses=6,
    motor_fanin=4,
)

# Create model using Keras 3.x
model = keras.Sequential([
    keras.layers.Input(shape=(None, 2)),
    keras.layers.RNN(
        LTC(ncp_wiring, cell_class=EnhancedLTCCell, solver="rk4", ode_unfolds=6),
        return_sequences=True
    )
])

# Use Keras 3.x training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.MeanSquaredError()
)

model.fit(data_x, data_y, batch_size=1, epochs=400)
