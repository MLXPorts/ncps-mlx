# Contributing to NCPS

Thank you for your interest in contributing to NCPS! This guide will help you get started with contributing to the project.

## Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/mlx-team/ncps-mlx.git
   cd ncps-mlx
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Write descriptive docstrings in Google format
- Keep functions and methods focused and concise

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting changes
- Add integration tests for complex features
- Include performance tests where relevant

### Documentation
- Update documentation for new features
- Include docstrings for all public APIs
- Add examples for complex functionality
- Update the changelog

## Pull Request Process

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

3. Push your changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Ensure your PR:
   - Passes all tests
   - Updates documentation as needed
   - Includes a changelog entry
   - Follows code style guidelines

## Code Review Process

1. All code changes require review
2. Address review feedback promptly
3. Keep PR scope focused and manageable
4. Be responsive to questions and comments

## Documentation Contributions

1. Set up documentation environment:
   ```bash
   cd docs
   ./tools/setup.sh
   source activate_docs
   ```

2. Make documentation changes
3. Build and test locally:
   ```bash
   doctools build
   doctools preview
   ```

4. Submit changes following the PR process

## Getting Help

- Join our community chat
- Check existing issues and discussions
- Ask questions in pull requests
- Review documentation

Thank you for contributing to NCPS!