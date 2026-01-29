# Contributing to Stochastic Robotics Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/stochastic-robotics-framework.git
cd stochastic-robotics-framework
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We follow PEP 8 style guidelines with the following tools:

- **Black**: Code formatting (line length: 100)
- **Flake8**: Linting
- **MyPy**: Type checking

Run formatting and checks:
```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
mypy src/
```

## Testing

We use `pytest` for testing. All new features should include tests.

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

Test guidelines:
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Aim for >80% code coverage
- Include both unit and integration tests

## Pull Request Process

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add: brief description of changes"
```

Commit message format:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements
- `Docs:` for documentation
- `Test:` for tests

3. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

4. PR Checklist:
   - [ ] Code follows style guidelines (Black, Flake8)
   - [ ] All tests pass
   - [ ] New tests added for new features
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated

## Areas for Contribution

### High Priority
- Additional noise models (Perlin noise, Brownian motion)
- More Kalman filter variants (UKF, EKF)
- ROS 2 integration examples
- Sim-to-real transfer case studies
- Performance optimizations

### Medium Priority
- Additional planning algorithms (RRT*, A* with uncertainty)
- More sophisticated domain randomization strategies
- Integration with Isaac Gym / Isaac Lab
- Vision-based noise models (blur, occlusion)
- Documentation improvements

### Beginner Friendly
- Adding examples for new environments
- Improving error messages
- Writing tutorials
- Creating visualization tools
- Bug fixes

## Documentation

Documentation is built with Sphinx. To build locally:
```bash
cd docs
make html
```

Documentation guidelines:
- Use Google-style docstrings
- Include examples in docstrings
- Add new modules to API reference
- Update README.md for user-facing changes

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email: cherniyassine@gomyrobot.com

Thank you for contributing! 🚀
