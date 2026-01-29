# Stochastic Robotics Framework - Project Structure

## Directory Organization

```
stochastic-robotics-framework/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── src/
│   └── stochastic_robotics/       # Main package
│       ├── __init__.py
│       ├── noise_models/          # Noise model implementations
│       │   ├── __init__.py
│       │   ├── base_noise.py      # Base class and scheduler
│       │   ├── gaussian.py        # Gaussian noise models
│       │   ├── ornstein_uhlenbeck.py  # OU process for correlated noise
│       │   └── salt_pepper.py     # Impulse and dropout noise
│       ├── filters/               # State estimation algorithms
│       │   ├── __init__.py
│       │   └── particle_filter.py # Particle filter (MCL)
│       ├── wrappers/              # Gymnasium environment wrappers
│       │   ├── __init__.py
│       │   └── noise_injection.py # Noise injection & domain randomization
│       └── evaluation/            # Robustness assessment tools
│           ├── __init__.py
│           └── mc_evaluator.py    # Monte Carlo evaluation
├── configs/
│   └── noise_config.yaml          # Default noise configuration
├── examples/
│   ├── demo.py                    # Quick demonstration
│   ├── train_robust_policy.py    # Full training script
│   └── evaluate_robustness.py    # Evaluation script
├── tests/
│   └── test_noise_models.py      # Unit tests
├── docs/                          # Documentation (to be added)
├── experiments/                   # Experimental results (to be added)
├── notebooks/                     # Jupyter notebooks (to be added)
├── README.md                      # Main readme
├── GETTING_STARTED.md            # Getting started guide
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                        # MIT License
├── setup.py                       # Package installation
├── requirements.txt               # Dependencies
├── pytest.ini                     # Pytest configuration
└── .gitignore                     # Git ignore rules
```

## Core Components

### 1. Noise Models (`src/stochastic_robotics/noise_models/`)

**Purpose**: Simulate realistic sensor and actuator imperfections

**Components**:
- `base_noise.py`: Abstract base class for all noise models
- `gaussian.py`: Gaussian white noise for sensors
- `ornstein_uhlenbeck.py`: Time-correlated noise for actuators
- `salt_pepper.py`: Impulse noise for cameras, dropout for sensors

**Key Classes**:
- `BaseNoise`: Abstract interface
- `GaussianNoise`: N(μ, σ²) noise
- `OrnsteinUhlenbeckNoise`: Correlated noise with mean reversion
- `SaltPepperNoise`: Impulse noise
- `DropoutNoise`: Random dropout
- `NoiseScheduler`: Dynamic noise scheduling during training

### 2. Filters (`src/stochastic_robotics/filters/`)

**Purpose**: State estimation under uncertainty

**Components**:
- `particle_filter.py`: Sequential Monte Carlo localization

**Key Classes**:
- `ParticleFilter`: SIR particle filter
- `AdaptiveParticleFilter`: AMCL with dynamic particle count

**Features**:
- Multiple resampling methods (systematic, multinomial, residual)
- Effective sample size monitoring
- Configurable motion and observation models

### 3. Wrappers (`src/stochastic_robotics/wrappers/`)

**Purpose**: Integrate noise with Gymnasium environments

**Components**:
- `noise_injection.py`: Noise injection and domain randomization wrappers

**Key Classes**:
- `NoiseInjectionWrapper`: Applies noise to observations and actions
- `DomainRandomizationWrapper`: Randomizes parameters at reset

**Features**:
- Modality-specific noise (joints, IMU, camera, etc.)
- YAML-based configuration
- Compatible with Stable-Baselines3
- Physics randomization support

### 4. Evaluation (`src/stochastic_robotics/evaluation/`)

**Purpose**: Statistical robustness assessment

**Components**:
- `mc_evaluator.py`: Monte Carlo evaluation tools

**Key Classes**:
- `MCEvaluator`: Policy evaluation across noise levels
- `EvaluationResult`: Data class for results

**Features**:
- Multi-episode evaluation
- Statistical metrics (mean, std, CI, percentiles)
- Visualization (robustness curves, distributions)
- Success rate tracking

## Configuration System

### Noise Configuration (`configs/noise_config.yaml`)

The framework uses YAML for configuration with the following structure:

```yaml
noise_injection:
  enabled: true
  mode: "domain_randomization"
  
  sensors:
    joint_position:
      type: "gaussian"
      sigma_range: [0.0, 0.1]
    # ... more sensors
  
  actuators:
    motor_commands:
      type: "ornstein_uhlenbeck"
      sigma_range: [0.0, 0.1]

physics_randomization:
  enabled: true
  dynamics:
    mass_range: [0.8, 1.2]
    friction_range: [0.5, 1.5]

training:
  algorithm: "PPO"
  total_timesteps: 10_000_000

evaluation:
  n_episodes: 100
  noise_levels: [0.0, 0.05, 0.1, 0.2]
```

## Examples

### Example Scripts (`examples/`)

1. **demo.py**: Quick demonstration of framework capabilities
   - Noise models showcase
   - Basic training
   - Robustness evaluation

2. **train_robust_policy.py**: Full-featured training script
   - Command-line arguments
   - Parallel environments
   - Domain randomization
   - Periodic evaluation
   - Model checkpointing

3. **evaluate_robustness.py**: Evaluation script
   - Load trained models
   - Test across noise levels
   - Generate plots and statistics

## Testing

### Test Structure (`tests/`)

- `test_noise_models.py`: Unit tests for all noise models
  - Initialization tests
  - Statistical property tests
  - Shape preservation tests
  - Configuration loading tests

**Running Tests**:
```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

## Development Workflow

### 1. Local Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/ examples/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### 2. Adding New Features

1. Create feature branch: `git checkout -b feature/new-noise-model`
2. Implement feature in appropriate module
3. Add unit tests in `tests/`
4. Update documentation
5. Run all checks (black, flake8, mypy, pytest)
6. Submit pull request

### 3. Adding New Noise Models

```python
# In src/stochastic_robotics/noise_models/my_noise.py
from .base_noise import BaseNoise

class MyNoise(BaseNoise):
    def __init__(self, param, seed=None):
        super().__init__(seed)
        self.param = param
    
    def apply(self, signal):
        # Your implementation
        return noisy_signal
    
    def reset(self):
        # Reset state
        pass
```

Then add to `__init__.py` and create tests.

### 4. Adding New Environments

```python
# Create custom config
custom_config = {
    'noise_injection': {
        'sensors': {
            'custom_sensor': {
                'type': 'gaussian',
                'sigma': 0.05
            }
        }
    }
}

# Wrap environment
env = gym.make("MyCustomEnv-v1")
env = NoiseInjectionWrapper(env, config=custom_config)
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

- **Triggers**: Push to main/develop, pull requests
- **Matrix Testing**: Python 3.8-3.11 on Ubuntu and macOS
- **Checks**: 
  - Linting (flake8)
  - Formatting (black)
  - Type checking (mypy)
  - Unit tests (pytest)
  - Coverage reporting (codecov)
  - Package building

## Future Extensions

### Planned Modules

1. **planning/** - Motion planning under uncertainty
   - MCTS implementation
   - RRT* with collision probability
   - Model predictive control

2. **visualization/** - Enhanced visualization tools
   - Real-time particle filter visualization
   - Policy rollout rendering
   - Workspace analysis plots

3. **ros2/** - ROS 2 integration
   - ROS 2 node wrappers
   - Message converters
   - Launch files

4. **benchmarks/** - Standardized benchmarks
   - Standard test environments
   - Baseline results
   - Leaderboards

## Dependencies

### Core Dependencies
- `numpy`: Numerical computing
- `scipy`: Scientific computing
- `gymnasium`: RL environment interface
- `mujoco`: Physics simulation
- `stable-baselines3`: RL algorithms
- `torch`: Deep learning

### Visualization
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `pandas`: Data manipulation

### Configuration & Logging
- `pyyaml`: Configuration files
- `tensorboard`: Training logs
- `wandb`: Experiment tracking (optional)

### Development
- `pytest`: Testing
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking

## Version History

- **v0.1.0** (2026-01): Initial release
  - Core noise models
  - Particle filter
  - Gymnasium wrappers
  - Monte Carlo evaluation
  - Basic documentation

## License

MIT License - See LICENSE file for details

## Contact

- **Author**: Yassine Cherni
- **Email**: cherniyassine@gomyrobot.com
- **GitHub**: https://github.com/yassine-cherni/stochastic-robotics-framework
