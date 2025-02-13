# Neural Circuit Policies (NCPs) with MLX

This repository provides efficient implementations of liquid neural networks using Apple's MLX framework. The implementation includes Closed-form Continuous-time (CfC) and Liquid Time-Constant (LTC) networks with a modular, extensible architecture.

## Features

- **Modular Architecture**: Base classes and mixins for easy extension and customization
- **Time-Aware Processing**: Handle variable time steps and continuous-time dynamics
- **Bidirectional Support**: Process sequences in both forward and backward directions
- **Backbone Networks**: Add feature extraction layers for enhanced representation learning
- **MLX Optimization**: Efficient implementation using MLX's lazy evaluation and automatic differentiation

## Installation

```bash
pip install ncps-mlx
```

## Quick Start

Here's a simple example using CfC for sequence processing:

```python
import mlx.core as mx
import mlx.nn as nn
from ncps.mlx import CfC

# Create a CfC model
model = CfC(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    bidirectional=True,
    return_sequences=True
)

# Process a sequence
x = mx.random.normal((batch_size, seq_length, input_size))
outputs, states = model(x)
```

For time-aware processing:

```python
# Create time deltas
time_delta = mx.ones((batch_size, seq_length, 1))

# Process with variable time steps
outputs, states = model(x, time_delta=time_delta)
```

## Architecture Overview

The implementation follows a modular design with several key components:

### Base Classes

- **LiquidCell**: Base class for liquid neuron cells (CfC, LTC)
- **LiquidRNN**: Base class for liquid neural networks

### Mixins

- **TimeAwareMixin**: Handles time-aware processing
- **BackboneMixin**: Manages backbone layers for feature extraction

### Implementations

- **CfC**: Closed-form Continuous-time networks
- **LTC**: Liquid Time-Constant networks

## Advanced Usage

Check out the example notebooks for advanced usage patterns:

- `examples/notebooks/mlx_cfc_example.ipynb`: CfC examples and benchmarks
- `examples/notebooks/mlx_ltc_rnn_example.ipynb`: LTC examples with time-aware processing

## Documentation

Comprehensive documentation is available in the `docs` directory:

- API Reference: `docs/api/mlx.rst`
- Examples: `docs/examples/`
- Quickstart Guide: `docs/quickstart.rst`

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

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
```

## Acknowledgments

- Original NCP implementation by Mathias Lechner and Ramin Hasani
- MLX port by Sydney Renee
