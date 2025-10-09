import numpy as np
import pytest


def test_passenger_example_runs():
    from examples import passenger_predictor_mlx as passenger

    metrics = passenger.run_experiment(epochs=2, plot=False)
    assert "train_rmse" in metrics and "test_rmse" in metrics


def test_temperature_example_runs():
    from examples import temperature_predictor_mlx as temperature

    metrics = temperature.run_experiment(epochs=2, plot=False)
    assert "train_rmse" in metrics and "test_rmse" in metrics


def test_currency_example_runs():
    from examples import currency_predictor_mlx as currency

    metrics = currency.run_experiment(epochs=5, plot=False)
    assert {
        "train_rmse",
        "val_rmse",
        "test_rmse",
    }.issubset(metrics)


def test_icra_lidar_example_runs():
    from examples import icra_lidar_mlx as lidar

    seq_len = 8
    train_x = np.random.normal(size=(10, seq_len, 1)).astype(np.float32)
    train_y = np.random.normal(size=(10, seq_len, 1)).astype(np.float32)
    test_x = np.random.normal(size=(4, seq_len, 1)).astype(np.float32)
    test_y = np.random.normal(size=(4, seq_len, 1)).astype(np.float32)
    metrics = lidar.run_experiment(epochs=1, batch_size=4, seq_len=seq_len, dataset=(train_x, train_y, test_x, test_y))
    assert "test_rmse" in metrics
