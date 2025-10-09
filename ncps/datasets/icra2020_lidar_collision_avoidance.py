"""ICRA 2020 lidar collision-avoidance dataset utilities (Apache-2.0)."""

from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np


def _augment_data(data):
    augmented = []
    for x, y in data:
        augmented.append((x, y))
        x_mirror = x[:, ::-1]
        y_mirror = -y
        augmented.append((x_mirror, y_mirror))
    return augmented


def _unpack(npz_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    with np.load(npz_path) as packed:
        for idx in range(29):
            sequences.append((packed[f"x_{idx}"], packed[f"y_{idx}"]))
    return sequences


def _train_test_split(files: list[tuple[np.ndarray, np.ndarray]]):
    rng = np.random.default_rng(20200822)
    rng.shuffle(files)
    return files[7:], files[:7]


def _align_in_sequences(data, seq_len: int):
    xs, ys = [], []
    for x, y in data:
        for start in range(0, x.shape[0] - seq_len, seq_len // 2):
            xs.append(x[start : start + seq_len])
            ys.append(y[start : start + seq_len])
    xs_arr = np.stack(xs, axis=0)
    ys_arr = np.stack(ys, axis=0)
    xs_arr = np.expand_dims(xs_arr, axis=-1)
    ys_arr = np.expand_dims(ys_arr, axis=-1)
    return xs_arr, ys_arr


def load_data(local_path: str | None = None, seq_len: int = 32):
    """Return (train, test) tuples of shape ``[N, T, 1]`` for x/y sequences."""

    url = "https://github.com/mlech26l/icra_lds/raw/master/icra2020_imitation_data_packed.npz"
    if local_path is None:
        os.makedirs("datasets", exist_ok=True)
        local_path = os.path.join("datasets", "icra2020_imitation_data_packed.npz")

    download = True
    if os.path.isfile(local_path):
        with open(local_path, "rb") as handle:
            md5 = hashlib.md5(handle.read()).hexdigest()
        if md5 == "15ab035e0866fc065acfc0ad781d75c5":
            download = False

    if download:
        with urllib.request.urlopen(url) as response, open(local_path, "wb") as out:
            out.write(response.read())

    raw = _unpack(local_path)
    train_raw, test_raw = _train_test_split(raw)
    train_aug = _augment_data(train_raw)
    test_aug = _augment_data(test_raw)

    return _align_in_sequences(train_aug, seq_len), _align_in_sequences(test_aug, seq_len)
