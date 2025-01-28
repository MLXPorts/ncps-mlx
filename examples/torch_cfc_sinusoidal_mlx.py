# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import ncps
import ncps.mini_keras as ks
from ncps import wirings
from ncps.mini_keras.layers import RNN

import ncps.mlx


# Custom LightningModule equivalent
class SequenceLearner:
    def __init__(self, model, lr=0.005):
        self.model = model
        self.optimizer = ks.optimizers.Adam(learning_rate=lr)
        self.loss_fn = ks.losses.MeanSquaredError()
        self.train_loss_metric = ks.metrics.Mean(name="train_loss")
        self.val_loss_metric = ks.metrics.Mean(name="val_loss")

    def training_step(self, x, y):
        with ks.GradientTape() as tape:
            y_hat = self.model(x, training=True)
            loss = self.loss_fn(y, y_hat)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss_metric.update_state(loss)
        return loss

    def validation_step(self, x, y):
        y_hat = self.model(x, training=False)
        loss = self.loss_fn(y, y_hat)
        self.val_loss_metric.update_state(loss)
        return loss

    def fit(self, train_dataset, val_dataset, epochs=10):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training loop
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                train_loss = self.training_step(x_batch, y_batch)

            print(f"Train Loss: {self.train_loss_metric.result().numpy():.4f}")

            # Validation loop
            for x_batch, y_batch in val_dataset:
                self.validation_step(x_batch, y_batch)

            print(f"Val Loss: {self.val_loss_metric.result().numpy():.4f}")

            # Reset metrics for the next epoch
            self.train_loss_metric.reset_states()
            self.val_loss_metric.reset_states()


# Define the models with ncps.mini_keras
def build_models(in_features, out_features):
    models = []
    models.append(
        ks.Sequential([
            ks.layers.Input(shape=(None, in_features)),  # Explicitly use shape= keyword
            ks.layers.RNN(
                ncps.mlx.LTCCell(
                    wiring=wirings.Random(128, output_dim=out_features, sparsity_level=0.5),
                    input_mapping="affine",
                    output_mapping="affine",
                    ode_unfolds=6,
                    dtype="float32"  # Use string dtype
                ),
                return_sequences=True,
            )
        ])
    )
    models.append(
        ks.Sequential([
            ks.layers.Input(shape=(None, in_features)),  # Explicitly use shape= keyword
            ks.layers.RNN(
                ks.WiredCfCCell(
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
                return_sequences=True,
            )
        ])
    )
    return models


# Example of single and stacked RNN usage
def create_example_models():
    # Single cell example
    cell = ncps.mlx.LTCCell(
        wiring=wirings.Random(128, output_dim=1, sparsity_level=0.5),
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        dtype="float32"  # Use string dtype
    )
    x = ncps.mini_keras.layers.Input(shape=(None, 5))
    single_layer = RNN(cell)
    y_single = single_layer(x)

    # Stacked cells example
    cells = [ncps.mlx.LTCCell(
        wiring=wirings.Random(128, output_dim=1, sparsity_level=0.5),
        input_mapping="affine",
        output_mapping="affine",
        ode_unfolds=6,
        dtype="float32"
    ) for _ in range(2)]  # Create 2 identical cells
    stacked_layer = RNN(cells)
    y_stacked = stacked_layer(x)
    
    return y_single, y_stacked


# Data Preparation
in_features = 2
out_features = 1
N = 48  # Length of the time-series
import mlx.core as mx
# Generate input features (sine and cosine wave)
data_x = mx.stack(
    [mx.sin(mx.linspace(0, 3 * mx.pi, N)), 
     mx.cos(mx.linspace(0, 3 * mx.pi, N))], 
    axis=1
)
# Add batch dimension and use MLX's Dtype
data_x = mx.expand_dims(data_x, axis=0).astype(mx.float32)

# Target output: sine wave with double the frequency
data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N)).reshape([1, N, 1]).astype(mx.float32)

# Don't convert to TensorFlow tensors, keep as MLX arrays
dataset = list(zip([data_x], [data_y]))  # Simple dataset as list of tuples
train_dataset = dataset
val_dataset = dataset

# Train models
for model in build_models(in_features, out_features):
    learner = SequenceLearner(model, lr=0.01)
    learner.fit(train_dataset, val_dataset, epochs=10)
