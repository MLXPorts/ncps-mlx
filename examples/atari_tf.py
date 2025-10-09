"""MLX CfC Atari demo mirroring the original TensorFlow example."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps_mlx import CfC
from ncps_mlx.wirings import FullyConnected


class Flatten(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=1)


class AtariCfC(nn.Module):
    def __init__(self, action_dim: int, rnn_units: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.rnn = CfC(
            128,
            FullyConnected(units=rnn_units, output_dim=rnn_units),
            return_sequences=True,
        )
        self.policy = nn.Linear(rnn_units, action_dim)
        self.value = nn.Linear(rnn_units, 1)

    def __call__(self, frames: mx.array):
        batch, time, height, width = frames.shape
        flat = mx.reshape(frames, (batch * time, height, width, 1))
        features = self.conv(flat)
        features = mx.reshape(features, (batch, time, -1))
        rnn_out, _ = self.rnn(features)
        logits = self.policy(rnn_out)
        value = self.value(rnn_out)
        return logits, value


def main() -> None:
    batch, time, h, w = 4, 32, 84, 84
    observations = mx.random.uniform(shape=(batch, time, h, w))
    actions = mx.random.randint(low=0, high=4, shape=(batch, time, 1)).astype(mx.float32)

    model = AtariCfC(action_dim=4)
    mx.eval(model.parameters())

    def loss_fn(mdl: AtariCfC, obs: mx.array, targets: mx.array):
        logits, values = mdl(obs)
        policy_loss = mx.mean((logits - targets) ** 2)
        value_loss = mx.mean(values ** 2)
        return policy_loss + 0.5 * value_loss

    value_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-3)

    for epoch in range(50):
        loss, grads = value_and_grad(model, observations, actions)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:02d} loss={float(loss.item()):.4f}")


if __name__ == "__main__":
    main()
