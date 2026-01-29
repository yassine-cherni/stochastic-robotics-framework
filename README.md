# Stochastic Robotics Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive framework for applying stochastic simulation and Monte Carlo methods to enhance the robustness and reliability of autonomous robotic systems.

## Features

- 🎲 **Monte Carlo Methods**: Particle filters, MCTS, workspace analysis
- 🔊 **Realistic Noise Models**: Gaussian, salt-and-pepper, Ornstein-Uhlenbeck processes
- 🤖 **RL Integration**: PPO and SAC with domain randomization
- 🎯 **MuJoCo Simulation**: Modular noise injection for realistic sim-to-real transfer
- 📊 **Statistical Analysis**: Comprehensive robustness evaluation tools
- 🔌 **ROS 2 Bridge**: Seamless deployment to real robots

## Installation

```bash
# Clone the repository
git clone https://github.com/yassine-cherni/stochastic-robotics-framework.git
cd stochastic-robotics-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

```python
from stochastic_robotics import NoiseInjectionWrapper, MCEvaluator
from stable_baselines3 import PPO
import gymnasium as gym

# Create environment with noise injection
env = gym.make("UnitreeGo1-v1")
env = NoiseInjectionWrapper(env, config="configs/noise_config.yaml")

# Train robust policy
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Evaluate robustness
evaluator = MCEvaluator(env, model)
results = evaluator.evaluate_robustness(
    noise_levels=[0.0, 0.05, 0.1, 0.2],
    n_episodes=100
)
evaluator.plot_robustness_curves(results)
```

## Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## Repository Structure

```
stochastic-robotics-framework/
├── src/
│   ├── noise_models/       # Noise model implementations
│   ├── filters/            # Particle filters and Kalman filters
│   ├── planning/           # MCTS and motion planning
│   ├── wrappers/           # Gymnasium wrappers
│   └── evaluation/         # Monte Carlo evaluation tools
├── configs/                # Configuration files
├── experiments/            # Experimental results
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
└── examples/               # Example scripts
```

## Citation

```bibtex
@software{cherni2026stochastic,
  author = {Cherni, Yassine},
  title = {Stochastic Simulation Framework for Robust Robotic Systems},
  year = {2026},
  url = {https://github.com/yassine-cherni/stochastic-robotics-framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Yassine Cherni - cherniyassine@gomyrobot.com

Project Link: [https://github.com/yassine-cherni/stochastic-robotics-framework](https://github.com/yassine-cherni/stochastic-robotics-framework)
