<div align="center"><img src="https://raw.githubusercontent.com/mlech26l/ncps/master/docs/img/banner.png" width="800"/></div>

# Neural Circuit Policies for Apple MLX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![pyversion](misc/pybadge.svg)
![GitHub stars](https://img.shields.io/github/stars/SolaceHarmony/ncps-mlx)
![GitHub issues](https://img.shields.io/github/issues/SolaceHarmony/ncps-mlx)
[![Documentation Status](https://readthedocs.org/projects/ncps-mlx/badge/?version=latest)](https://ncps-mlx.readthedocs.io/en/latest/?badge=latest)

## 🎯 About This Project

This is a **complete MLX rewrite** of Neural Circuit Policies by **Sydney Renee** for **The Solace Project**. This implementation is specifically optimized for Apple Silicon using Apple's MLX framework, providing native GPU acceleration on M-series chips.

### Original Research

Neural Circuit Policies (NCPs) are designed sparse recurrent neural networks loosely inspired by the nervous system of [C. elegans](http://www.wormbook.org/chapters/www_celegansintro/celegansintro.html):

- [Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf)
- [Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)

### Attribution

This project is based on the original [ncps](https://github.com/mlech26l/ncps) library by Mathias Lechner and contributors, which provided PyTorch and TensorFlow implementations. This MLX version is a ground-up rewrite specifically for Apple Silicon.

## 🚀 Quick Start

```python
import mlx.core as mx
from ncps import CfC
from ncps.wirings import AutoNCP

# Create a CfC model with 20 input features and 50 hidden units
rnn = CfC(input_size=20, units=50)

# Or use structured NCP wiring
wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
rnn = CfC(input_size=20, units=wiring)

# Forward pass
x = mx.random.normal((2, 3, 20))  # (batch, time, features)
output, state = rnn(x)
```


## 📦 Installation

### Requirements
- **Apple Silicon Mac** (M1, M2, M3, M4, or later)
- **Python 3.8+**
- **MLX 0.1.0+**

### Install from Source

```bash
# Clone the repository
git clone https://github.com/SolaceHarmony/ncps-mlx.git
cd ncps-mlx

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e .[viz]  # Includes matplotlib and networkx for visualization
```

### Install MLX (if not already installed)

```bash
pip install mlx
```

## 📚 Documentation

Full documentation is available at [ReadTheDocs](https://ncps-mlx.readthedocs.io/) (coming soon) or can be built locally:

```bash
cd docs
pip install -r ../.readthedocs-requirements.txt
make html
# Open docs/_build/html/index.html in your browser
```

The documentation includes:
- **[Quickstart Guide](docs/quickstart.rst)** - Get started with NCPs and MLX
- **[API Reference](docs/api/index.rst)** - Complete API documentation
- **[Examples](docs/examples/index.rst)** - Detailed usage examples and tutorials

## 📚 Examples

Check out the [examples](./examples) directory for complete working examples:

- **[mlx_smnist_training.py](./examples/mlx_smnist_training.py)** - Sequential MNIST classification with LTC
- **[mlx_cfc_regression.py](./examples/mlx_cfc_regression.py)** - Time series regression with CfC
- **[currency_predictor_mlx.py](./examples/currency_predictor_mlx.py)** - Currency prediction example
- **[maze_train_mlx.py](./examples/maze_train_mlx.py)** - Maze navigation training
- **[mlx_cell_comparison.py](./examples/mlx_cell_comparison.py)** - Compare different RNN cell types

### Running Examples

```bash
# Train on Sequential MNIST
python examples/mlx_smnist_training.py --epochs 200 --hidden-size 64

# Time series regression
python examples/mlx_cfc_regression.py
```

## 🧠 Models and Wirings

This package provides MLX implementations of liquid time-constant (LTC) and closed-form continuous-time (CfC) neural networks as `mlx.nn.Module` layers.

### Available Models

```python
from ncps import CfC, LTC, CTRNN, CTGRU
from ncps.wirings import AutoNCP, FullyConnected

# Fully-connected models
input_size = 20
units = 28  # 28 neurons

rnn = CfC(input_size, units)
rnn = LTC(input_size, units)
rnn = CTRNN(input_size, units)
rnn = CTGRU(input_size, units)
```

### Structured NCP Wirings

The key innovation of NCPs is their structured wiring diagrams, inspired by biological neural circuits. You can use predefined wiring patterns:

```python
from ncps import CfC, LTC
from ncps.wirings import AutoNCP, FullyConnected

# AutoNCP: Automatically generates a structured NCP wiring
wiring = AutoNCP(28, 4)  # 28 neurons, 4 outputs
input_size = 20

rnn = CfC(input_size, wiring)
rnn = LTC(input_size, wiring)
```

![NCP Wiring Diagram](https://github.com/mlech26l/ncps/raw/master/docs/img/things.png)

### Training Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from ncps import LTC
from ncps.wirings import FullyConnected

# Create model
wiring = FullyConnected(units=64, output_dim=64)
wiring.build(input_size=20)
model = LTC(input_size=20, units=wiring, return_sequences=True)


# Define loss function
def loss_fn(model, x, y):
    outputs, _ = model(x)
    return mx.mean((outputs - y) ** 2)


# Training loop
optimizer = optim.Adam(learning_rate=1e-3)
value_and_grad_fn = nn.value_and_grad(model, loss_fn)

for epoch in range(100):
    loss, grads = value_and_grad_fn(model, x_train, y_train)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

## 🔄 Key Differences from Original

This MLX implementation differs from the original PyTorch/TensorFlow version:

### Architecture Changes
- **MLX-Native Operations**: All operations use MLX primitives for optimal Apple Silicon performance
- **Unified API**: Single, consistent API instead of separate torch/tf modules
- **Direct Import**: `from ncps import LTC, CfC` (no need for `ncps.mlx`)
- **Batch-First by Default**: Follows MLX conventions with `(batch, time, features)` ordering

### Performance Optimizations
- **Apple Neural Engine**: Automatically leverages Apple's specialized ML hardware
- **Unified Memory**: Takes advantage of Apple Silicon's unified memory architecture
- **Metal Backend**: Uses Metal for GPU acceleration on M-series chips

### API Improvements
- **Simpler Initialization**: More intuitive parameter naming
- **Better Type Hints**: Full type annotations for better IDE support
- **State Management**: Cleaner hidden state handling
- **Model Checkpointing**: Native support for MLX weight serialization

### Example: Creating a Sequential Model

```python
import mlx.core as mx
import mlx.nn as nn
from ncps import CfC
from ncps.wirings import AutoNCP


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        wiring = AutoNCP(hidden_dim, num_classes)
        self.rnn = CfC(input_size=input_dim, units=wiring, return_sequences=False)
        self.output = nn.Linear(self.rnn.output_size, num_classes)

    def __call__(self, x):
        features, _ = self.rnn(x)
        return self.output(features)


# Create and use model
model = SequenceClassifier(input_dim=20, hidden_dim=64, num_classes=10)
x = mx.random.normal((32, 100, 20))  # (batch, time, features)
logits = model(x)
```

## ✨ Features

- **🚀 Optimized for Apple Silicon**: Native MLX implementation for M1/M2/M3/M4 chips
- **🧠 Multiple RNN Architectures**: LTC, CfC, CTRNN, CTGRU, and more
- **🔌 Flexible Wiring**: Support for structured NCP wirings and fully-connected layers
- **📊 Production Ready**: Includes model checkpointing, state management, and profiling
- **🎯 Type Safe**: Full type annotations for better development experience
- **📈 Easy Training**: Compatible with standard MLX training patterns
- **🔧 Extensible**: Easy to customize and extend for research

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this software in your research, please cite both the original NCP papers and this implementation:

### Original NCP Research
```bibtex
@article{lechner2020neural,
  title={Neural circuit policies enabling auditable autonomy},
  author={Lechner, Mathias and Hasani, Ramin and Amini, Alexander and Henzinger, Thomas A and Rus, Daniela and Grosu, Radu},
  journal={Nature Machine Intelligence},
  volume={2},
  number={10},
  pages={642--652},
  year={2020},
  publisher={Nature Publishing Group}
}

@article{hasani2021closed,
  title={Closed-form continuous-time neural networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Liebenwein, Lucas and Ray, Aaron and Tschaikowski, Max and Teschl, Gerald and Rus, Daniela},
  journal={Nature Machine Intelligence},
  volume={4},
  number={11},
  pages={992--1003},
  year={2022},
  publisher={Nature Publishing Group}
}
```

### This Implementation
```bibtex
@software{ncps_mlx_2025,
  title={ncps-mlx: Neural Circuit Policies for Apple MLX},
  author={Renee, Sydney},
  year={2025},
  url={https://github.com/SolaceHarmony/ncps-mlx},
  note={MLX implementation for Apple Silicon}
}
```

## 🔗 Links

- **GitHub Repository**: [https://github.com/SolaceHarmony/ncps-mlx](https://github.com/SolaceHarmony/ncps-mlx)
- **Original NCP Repository**: [https://github.com/mlech26l/ncps](https://github.com/mlech26l/ncps)
- **Apple MLX**: [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

## 🙏 Acknowledgments

This project builds upon the groundbreaking work of Mathias Lechner, Ramin Hasani, and their colleagues on Neural Circuit Policies. We are grateful for their research and the original open-source implementation that made this MLX port possible.

---

**Developed by Sydney Renee for The Solace Project** 🌟
