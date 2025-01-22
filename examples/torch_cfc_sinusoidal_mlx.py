# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import tensorflow as tf
import ncps
import ncps.mini_keras as ks
from ncps import wirings
import mlx.core as np

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
    return [
        ncps.mlx.LTCCell(wiring=list,
                         input_mapping="relu",
                         output_mapping="affine",
                         ode_unfolds=6,
                         in_features=in_features,
                         out_features=out_features),
        ks.Sequential([
            ks.layers.Input(shape=(None, in_features)),
            ks.layers.RNN(
                ks.layers.RNN(wiring=wirings.FullyConnected(8, out_features)),
                return_sequences=True,
            )
        ]),
        ks.Sequential([
            ks.layers.Input(shape=(None, in_features)),
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
    ]


# Data Preparation
in_features = 2
out_features = 1
N = 48  # Length of the time-series

# Generate input features (sine and cosine wave)
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output: sine wave with double the frequency
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)

# Convert MLX arrays to TensorFlow tensors
tf_data_x = tf.convert_to_tensor(data_x.tolist())
tf_data_y = tf.convert_to_tensor(data_y.tolist())

# Create TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices((tf_data_x, tf_data_y))
train_dataset = dataset.batch(1).shuffle(10)
val_dataset = dataset.batch(1)

# Train models
for model in build_models(in_features, out_features):
    learner = SequenceLearner(model, lr=0.01)
    learner.fit(train_dataset, val_dataset, epochs=10)