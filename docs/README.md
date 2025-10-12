# Documentation for ncps-mlx

This directory contains the Sphinx documentation for Neural Circuit Policies for Apple MLX.

## Building the Documentation

### Prerequisites

Install the required dependencies:

```bash
pip install -r ../.readthedocs-requirements.txt
```

Or install the minimal requirements:

```bash
pip install sphinx sphinx-copybutton sphinx-design furo
```

### Build HTML Documentation

```bash
make html
```

The generated HTML files will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Clean Build Files

```bash
make clean
```

## Documentation Structure

```
docs/
├── index.rst              # Main documentation page
├── quickstart.rst         # Quickstart guide for NCPs
├── conf.py               # Sphinx configuration
├── api/                  # API reference documentation
│   ├── index.rst
│   ├── mlx.rst          # MLX models documentation
│   └── wirings.rst      # Wiring patterns documentation
├── examples/             # Usage examples and tutorials
│   ├── index.rst
│   ├── mlx_quickstart.rst   # Quick start examples
│   └── mlx_training.rst     # Training examples
├── img/                  # Images and diagrams
├── _static/             # Static files (CSS, etc.)
├── Makefile             # Unix build file
└── make.bat             # Windows build file
```

## Documentation Content

The documentation is adapted from the original [ncps](https://github.com/mlech26l/ncps) library by Mathias Lechner, with updates specific to the MLX implementation:

- **MLX-specific API**: All code examples use MLX instead of PyTorch/TensorFlow
- **Apple Silicon optimization**: Documentation highlights Apple Silicon features
- **Unified API**: Single API instead of separate torch/tf modules
- **Training examples**: Complete MLX training patterns and examples

## Contributing to Documentation

When adding new features or making changes to the library:

1. Update relevant `.rst` files in `docs/`
2. Add code examples demonstrating the new features
3. Build the docs locally to verify changes: `make html`
4. Ensure no warnings or errors in the build output

### Documentation Style Guide

- Use clear, concise language
- Include code examples for all features
- Explain the "why" not just the "what"
- Link to relevant papers or resources
- Use proper RST formatting

## ReadTheDocs Integration

This documentation is configured to build automatically on [ReadTheDocs](https://readthedocs.org/) when changes are pushed to the repository.

Configuration files:
- `.readthedocs.yml` - ReadTheDocs configuration
- `.readthedocs-requirements.txt` - Documentation dependencies

## License

The documentation follows the same Apache 2.0 license as the main project.
