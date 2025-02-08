# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)

from ncps import mini_keras as mini_keras
from ncps.mini_keras import ops
from ncps import wirings
from ncps.mlx import CfC, WiredCfCCell
from ncps.mini_keras.models import Sequential, Model
# Define the sequence model using mini_keras Model API 

# Data preparation using Keras ops
N = 48  # Length of the time-series
in_features = 2
out_features = 1

# Input feature is a sine and a cosine wave using Keras ops
t = ops.linspace(0, 3 * ops.pi(), N)
sin_wave = ops.sin(t)
cos_wave = ops.cos(t)
data_x = ops.stack([sin_wave, cos_wave], axis=1)
data_x = ops.expand_dims(data_x, axis=0)  # Add batch dimension

# Target output is a sine with double the frequency
t_out = ops.linspace(0, 6 * ops.pi(), N)  
data_y = ops.sin(t_out)
data_y = ops.reshape(data_y, [1, N, 1])

# List of models to test
model_configs = [
    CfC(units=32),
    WiredCfCCell(
        wiring=wirings.FullyConnected(8, out_features)
    ),
    WiredCfCCell(
        wiring=wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        )
    ),
    CfC(
        units=32,
        mode="pure"
    ),
    WiredCfCCell(
        wiring=wirings.FullyConnected(8, out_features),
        mode="pure"
    ),
    WiredCfCCell(
        wiring=wirings.NCP(
            inter_neurons=16,
            command_neurons=8,
            motor_neurons=out_features,
            sensory_fanout=12,
            inter_fanout=4,
            recurrent_command_synapses=5,
            motor_fanin=8,
        ),
        mode="pure"
    ),
]


# Approach 2: Using standard Keras training (model.compile and model.fit)
# Comment out or remove the SupervisedTrainer related code above
for rnn_cell in model_configs:
    # Use proper Keras model building pattern
    
    inputs = mini_keras.Input(shape=(N, in_features))
    x = mini_keras.layers.RNN(rnn_cell, return_sequences=True)(inputs)
    model = mini_keras.Model(inputs=inputs, outputs=x)

    model.compile(optimizer=mini_keras.optimizers.Adam(learning_rate=0.01),
                 loss='mse',
                 metrics=['mse'])
    
    print(f"\nTraining model: {rnn_cell.__class__.__name__}")

    history = model.fit(data_x.astype("float32"), data_y.astype("float32"), 
                            batch_size=1, epochs=10, verbose=1)
    
    loss, mse = model.evaluate(data_x, data_y)
    print(f"\nFinal training metrics for {rnn_cell.__class__.__name__}:")
    print(f"loss: {loss}")
    print(f"mse: {mse}")
