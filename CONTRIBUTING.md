# Contributing to ncps-mlx

Thank you for your interest in contributing to ncps-mlx! This document provides guidelines and instructions for contributing.

## 🌟 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues. When creating a bug report, include:

- A clear, descriptive title
- Exact steps to reproduce the problem
- Expected vs. actual behavior
- Code snippets (minimal reproducible example)
- Your environment (OS, chip, Python/MLX versions)
- Error messages and stack traces

### Suggesting Features

Feature requests are welcome! Please:

- Use a clear, descriptive title
- Provide a detailed description of the proposed feature
- Explain why this feature would be useful
- Consider if you can help implement it

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our style guidelines
3. **Test thoroughly** on Apple Silicon if possible
4. **Run the linter**: `python misc/emberlint.py ncps/mlx`
5. **Update documentation** if needed
6. **Write good commit messages** (see below)
7. **Submit the pull request**

## 🎨 Style Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to all public functions/classes
- Keep functions focused and concise
- Prefer MLX operations over NumPy where possible

### Docstring Format

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Start with a type prefix:
  - `feat:` New feature
  - `fix:` Bug fix
  - `docs:` Documentation changes
  - `style:` Code style changes (formatting, etc.)
  - `refactor:` Code refactoring
  - `test:` Adding or updating tests
  - `chore:` Maintenance tasks

Example: `feat: add support for dynamic time steps in CfC`

## 🧪 Testing

### Running Tests

```bash
# Install dev dependencies
pip install -e .[dev]

# Run all tests
pytest ncps/mlx/ops/tests/

# Run specific test file
pytest ncps/mlx/ops/tests/test_ops.py

# Run with coverage
pytest --cov=ncps.mlx
```

### Writing Tests

- Write tests for new features
- Ensure tests pass on both CPU and GPU (if available)
- Use descriptive test names
- Test edge cases and error conditions

## 🔍 Code Quality

### Linting

We use NCPSLint for code quality checks:

```bash
# Run full lint
python misc/emberlint.py ncps/mlx

# Check specific issues
python misc/emberlint.py ncps/mlx --numpy-only
python misc/emberlint.py ncps/mlx --unused-only
```

### Pre-commit Checks

Before committing:

1. ✅ Run linter
2. ✅ Run tests
3. ✅ Check code formatting
4. ✅ Update documentation
5. ✅ Review your changes

## 📚 Documentation

### Updating Documentation

- Update docstrings for any changed functions
- Add examples for new features
- Update README.md if adding major features
- Consider adding a notebook tutorial for complex features

### Documentation Style

- Be clear and concise
- Include code examples
- Explain the "why" not just the "what"
- Link to relevant papers or resources

## 🏗️ Development Setup

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ncps-mlx.git
cd ncps-mlx

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Set up pre-commit hooks (optional)
pre-commit install
```

### Project Structure

```
ncps-mlx/
├── ncps/
│   └── mlx/          # Main MLX implementation
│       ├── base.py   # Base classes
│       ├── cfc.py    # CfC model
│       ├── ltc.py    # LTC model
│       ├── wirings.py # Wiring patterns
│       └── ops/      # MLX operations
├── examples/         # Example scripts
├── docs/            # Documentation
├── tests/           # Test files
└── misc/            # Utilities (linter, etc.)
```

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different perspectives
- Help others learn and grow

### Getting Help

- 💬 Use GitHub Discussions for questions
- 🐛 Use GitHub Issues for bugs
- 📧 Email for sensitive matters
- 🔍 Search existing issues before creating new ones

## 📝 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## 🙏 Thank You!

Every contribution helps make ncps-mlx better. Whether it's:

- Reporting a bug
- Fixing a typo
- Adding a feature
- Improving documentation
- Helping other users

Your contribution is valuable and appreciated! 🎉

---

Questions? Open a discussion or reach out to the maintainers!
