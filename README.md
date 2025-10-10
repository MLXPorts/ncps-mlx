# NCPS-MLX

Pure MLX implementation of Neural Circuit Policies with CfC/LTC cells and sparse wiring.

## Installation

```bash
pip install -e .
```

## Tests

```bash
pytest
```

## Examples

See `examples/` for sine-wave regression, policy distillation, Atari prototypes, and more. Recent MLX refactors include:

- `examples/passenger_predictor_mlx.py` — airline passenger forecasting with LTC wiring. Use `run_experiment(epochs=300)` or run the script directly for full training.
- `examples/temperature_predictor_mlx.py` — daily minimum temperature regression with an LTC forecaster.
- `examples/currency_predictor_mlx.py` — USD/TRY currency prediction using batched LTC sequences and the archived USDTRY dataset (`examples/data/USDTRY_Data.csv`).
- `examples/stock_predictor_mlx.py` — Apple stock price regression driven by the ported LTC4 workflow (`--symbol`, `--epochs` flags available).

Example invocation:

```bash
PYTHONPATH=. python examples/passenger_predictor_mlx.py
```
