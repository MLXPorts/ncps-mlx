"""PPO-style RL for the tile maze with MLX + CfC policy.

This is a compact, self-contained PPO trainer (single env, on-policy) that
uses a CfC backbone to compute a Gaussian policy over continuous steering and
an MSE value head. Rewards encourage forward progress and clearance; collisions
are heavily penalized. Artifacts are saved to artifacts/maze_ppo.

Usage:
  PYTHONPATH=. python examples/maze_rl_ppo_mlx.py
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ncps import CfC, wirings
from examples.wiring_presets import make_sensory_motor_wiring
from examples.maze_env import MAP_ASCII, TILE, is_wall, raycast


ART = "artifacts/maze_ppo"


@dataclass
class PPOCfg:
    bins: int = 181
    fov: float = math.radians(270)
    lidar_max: float = TILE * 12
    speed: float = 120.0
    dt: float = 0.1
    gamma: float = 0.98
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 8
    steps_per_epoch: int = 2048
    minibatch: int = 256
    updates: int = 10


class CfCActorCritic(nn.Module):
    """Actor-Critic with an NCPS motor neuron for the policy head.

    - Actor: CfC with an NCPS wiring that exposes a single motor neuron
      (output_dim=1). Its readout is the mean action (mu) in [-1, 1] after tanh.
    - Critic: a compact CfC that projects to a scalar value per timestep.
    """

    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        # Actor with explicit motor neuron via wiring
        wiring = make_sensory_motor_wiring(input_dim=input_dim, units=hidden, output_dim=1)
        self.actor = CfC(
            input_size=input_dim,
            units=wiring,                # NCPS wiring; output_size == 1 (motor)
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            mode="default",
            activation="lecun_tanh",
        )
        # Critic: standalone CfC with projection to 1
        self.critic = CfC(
            input_size=input_dim,
            units=hidden,
            proj_size=1,
            return_sequences=True,
            batch_first=True,
            mode="default",
            activation="lecun_tanh",
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0.0,
        )
        # Log std as a free parameter for the Gaussian policy
        self.log_std = mx.zeros((1,), dtype=mx.float32)

    def __call__(self, x: mx.array, hx: mx.array | None = None):
        # For simplicity, do not share state between actor and critic
        mu, _ = self.actor(x, hx=None)           # [B,T,1]
        v,  _ = self.critic(x, hx=None)          # [B,T,1]
        mu = mx.tanh(mu)                         # clamp to [-1, 1]
        return mu, v, hx


class MazeEnv:
    def __init__(self, cfg: PPOCfg):
        self.cfg = cfg
        self.reset()

    def reset(self) -> mx.array:
        self.rx, self.ry = TILE * 2.5, TILE * (len(MAP_ASCII) - 2.5)
        self.heading = -math.pi / 2
        return self._obs()

    def _obs(self) -> mx.array:
        dists = []
        ang0 = -self.cfg.fov / 2
        for i in range(self.cfg.bins):
            a = self.heading + ang0 + self.cfg.fov * (i / (self.cfg.bins - 1))
            d = raycast(self.rx, self.ry, a, self.cfg.lidar_max)
            dists.append(min(1.0, d / self.cfg.lidar_max))
        v = self.cfg.speed / (TILE * 4)
        obs = mx.array([dists + [v, 0.0]], dtype=mx.float32)  # [1,D]
        return obs

    def step(self, steer: float) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        # Apply motion
        h_prev = self.heading
        self.heading += steer * 1.2 * self.cfg.dt
        nx = self.rx + math.cos(self.heading) * self.cfg.speed * self.cfg.dt
        ny = self.ry + math.sin(self.heading) * self.cfg.speed * self.cfg.dt
        collided = is_wall(nx, ny)
        if not collided:
            self.rx, self.ry = nx, ny

        # Reward shaping: forward progress projected along heading, clearance, penalties
        forward = self.cfg.speed * self.cfg.dt if not collided else 0.0
        # Front clearance
        ang0 = -self.cfg.fov / 2
        mid = self.cfg.bins // 2
        front = raycast(self.rx, self.ry, self.heading, self.cfg.lidar_max)
        clearance = min(1.0, front / self.cfg.lidar_max)
        r = 0.01 * forward + 0.5 * clearance - 0.01 * (steer * steer)
        if collided:
            r += -1.0

        obs = self._obs()
        done = mx.array(1 if collided else 0, dtype=mx.int32)
        rew = mx.array(r, dtype=mx.float32)
        return obs, rew, done, mx.array(0, dtype=mx.int32)


def rollout(env: MazeEnv, net: CfCActorCritic, cfg: PPOCfg):
    obs_buf = []
    act_buf = []
    logp_buf = []
    val_buf = []
    rew_buf = []
    done_buf = []
    hx = mx.zeros((1, 64), dtype=mx.float32)
    obs = env.reset()
    for t in range(cfg.steps_per_epoch):
        x = mx.expand_dims(obs, 0)  # [1,1,D]
        mu, v, hx = net(x, hx=hx)
        std = mx.exp(net.log_std)[0]
        # Sample action from Normal(mu, std)
        eps = mx.random.normal(shape=mu.shape)
        a = mx.clip(mu + std * eps, -1.0, 1.0)
        # Log prob (scalar per step)
        logp = -0.5 * (((a - mu) / (std + 1e-6)) ** 2 + 2 * mx.log(std) + mx.log(2 * mx.array(math.pi)))
        logp = mx.sum(logp, axis=-1)  # [1,1,1] -> [1,1]

        # Step env
        obs2, r, d, _ = env.step(float(a[0, 0, 0].tolist()))

        obs_buf.append(obs[0])
        act_buf.append(a[0, 0])
        val_buf.append(v[0, -1])
        logp_buf.append(logp[0, -1])
        rew_buf.append(r)
        done_buf.append(d)

        obs = obs2
        if int(d.tolist()) == 1:
            obs = env.reset()
            hx = mx.zeros((1, 64), dtype=mx.float32)

    # Stack
    OBS = mx.stack(obs_buf, axis=0)          # [T, D]
    ACT = mx.stack(act_buf, axis=0)          # [T, 1]
    LOGP = mx.stack(logp_buf, axis=0)        # [T]
    VAL = mx.stack(val_buf, axis=0)          # [T, 1]
    REW = mx.stack(rew_buf, axis=0)          # [T]
    DONE = mx.stack(done_buf, axis=0)        # [T]
    return OBS, ACT, LOGP, VAL, REW, DONE


def compute_gae(rew, val, done, cfg: PPOCfg):
    T = int(rew.shape[0])
    adv = mx.zeros((T,), dtype=mx.float32)
    lastgaelam = mx.array(0.0, dtype=mx.float32)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - mx.array(float(done[t]), dtype=mx.float32)
        delta = rew[t] + cfg.gamma * nonterminal * (val[t + 1] if t + 1 < T else 0.0) - val[t]
        lastgaelam = delta + cfg.gamma * cfg.lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + val
    # Normalize advantages
    adv = (adv - mx.mean(adv)) / (mx.std(adv) + 1e-8)
    return adv, ret


def train():  # pragma: no cover
    cfg = PPOCfg()
    os.makedirs(ART, exist_ok=True)
    env = MazeEnv(cfg)
    obs_dim = cfg.bins + 2
    net = CfCActorCritic(obs_dim, hidden=64)
    opt = optim.Adam(learning_rate=cfg.lr)

    def loss_batch(OBS, ACT, LOGP_OLD, ADV, RET):
        # Shape into [B,T,D]
        x = mx.reshape(OBS, (OBS.shape[0], 1, OBS.shape[1]))
        mu, v, _ = net(x, hx=None)
        mu = mu[:, -1, :]  # [B,1]
        v  = v[:, -1, 0]   # [B]
        std = mx.exp(net.log_std)[0]
        # Log prob under new policy
        logp = -0.5 * (((ACT - mu) / (std + 1e-6)) ** 2 + 2 * mx.log(std) + mx.log(2 * mx.array(math.pi)))
        logp = mx.sum(logp, axis=-1)
        ratio = mx.exp(logp - LOGP_OLD)
        surr1 = ratio * ADV
        surr2 = mx.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * ADV
        pi_loss = -mx.mean(mx.minimum(surr1, surr2))
        vf_loss = mx.mean((RET - v) ** 2)
        ent = 0.5 * (1.0 + mx.log(2 * mx.array(math.pi)) + 2 * mx.log(std))
        ent = mx.mean(ent)
        loss = pi_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent
        return loss

    value_and_grad = nn.value_and_grad(net, loss_batch)

    for upd in range(1, cfg.updates + 1):
        OBS, ACT, LOGP, VAL, REW, DONE = rollout(env, net, cfg)
        ADV, RET = compute_gae(REW, VAL[:, 0], DONE, cfg)
        # Create minibatches
        N = int(OBS.shape[0])
        idx = mx.random.permutation(N)
        for ep in range(cfg.epochs):
            for s in range(0, N, cfg.minibatch):
                sl = idx[s:s + cfg.minibatch]
                loss, grads = value_and_grad(OBS[sl], ACT[sl], LOGP[sl], ADV[sl], RET[sl])
                opt.update(net, grads)
                mx.eval(net.parameters(), opt.state)
        print("update", upd, "loss=", loss)

    # Save
    net.save_weights(os.path.join(ART, "weights.npz"))


if __name__ == "__main__":  # pragma: no cover
    train()
