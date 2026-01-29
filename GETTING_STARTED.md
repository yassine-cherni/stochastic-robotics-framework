# Getting Started with Stochastic Robotics Framework

This guide will help you get started with the framework through practical examples.

## Installation

### Basic Installation

```bash
git clone https://github.com/yassine-cherni/stochastic-robotics-framework.git
cd stochastic-robotics-framework
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### With ROS 2 Support

```bash
pip install -e ".[ros2]"
```

## Quick Start: 5-Minute Tutorial

### 1. Run the Demo

```bash
cd examples
python demo.py
```

This will demonstrate:
- Different noise models
- Training with noise injection
- Robustness evaluation

### 2. Train Your First Robust Policy

```bash
python train_robust_policy.py \
    --env-id Ant-v4 \
    --algo PPO \
    --domain-randomization \
    --total-timesteps 1000000 \
    --n-envs 8
```

### 3. Evaluate Robustness

```bash
python evaluate_robustness.py \
    --model results/stochastic_exp/final_model.zip \
    --env-id Ant-v4 \
    --n-episodes 100
```

## Core Concepts

### 1. Noise Models

The framework provides several noise models to simulate sensor and actuator imperfections:

```python
from stochastic_robotics.noise_models import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    SaltPepperNoise
)

# Gaussian white noise (for sensors)
sensor_noise = GaussianNoise(mu=0.0, sigma=0.1)
noisy_reading = sensor_noise.apply(clean_reading)

# Time-correlated noise (for actuators)
actuator_noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2)
noisy_command = actuator_noise.apply(command)

# Impulse noise (for cameras)
camera_noise = SaltPepperNoise(salt_prob=0.01, pepper_prob=0.01)
noisy_image = camera_noise.apply(clean_image)
```

### 2. Environment Wrappers

Wrap your Gymnasium environment with noise injection:

```python
import gymnasium as gym
from stochastic_robotics.wrappers import NoiseInjectionWrapper

env = gym.make("Ant-v4")
env = NoiseInjectionWrapper(
    env,
    config="configs/noise_config.yaml",
    observation_noise=True,
    action_noise=True
)
```

### 3. Domain Randomization

Enable domain randomization for robust policy learning:

```python
from stochastic_robotics.wrappers import DomainRandomizationWrapper

env = DomainRandomizationWrapper(
    env,
    config="configs/noise_config.yaml"
)
# Noise parameters are randomized at each episode reset
```

### 4. Monte Carlo Evaluation

Evaluate policy robustness statistically:

```python
from stochastic_robotics.evaluation import MCEvaluator

evaluator = MCEvaluator(env, policy, confidence_level=0.95)

results = evaluator.evaluate_robustness(
    noise_levels=[0.0, 0.05, 0.1, 0.2],
    n_episodes=100
)

evaluator.print_statistics(results)
evaluator.plot_robustness_curves(results)
```

## Configuration

### Noise Configuration (YAML)

Edit `configs/noise_config.yaml` to customize noise parameters:

```yaml
noise_injection:
  sensors:
    joint_position:
      type: "gaussian"
      sigma_range: [0.0, 0.1]  # Domain randomization range
    
    imu_accel:
      type: "gaussian"
      sigma_range: [0.0, 0.3]
  
  actuators:
    motor_commands:
      type: "ornstein_uhlenbeck"
      theta: 0.15
      sigma_range: [0.0, 0.1]

physics_randomization:
  enabled: true
  dynamics:
    mass_range: [0.8, 1.2]
    friction_range: [0.5, 1.5]
```

## Common Use Cases

### Use Case 1: Legged Locomotion

Train a quadruped robot with realistic sensor noise:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stochastic_robotics.wrappers import (
    NoiseInjectionWrapper,
    DomainRandomizationWrapper
)

# Create environment
env = gym.make("UnitreeGo1-v1")  # Example, may need custom env

# Add noise injection
env = NoiseInjectionWrapper(env, config="configs/quadruped_config.yaml")
env = DomainRandomizationWrapper(env, config="configs/quadruped_config.yaml")

# Train
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000_000)
```

### Use Case 2: Manipulation

Train a robotic arm with vision and proprioception noise:

```python
# Configure noise for joint encoders and camera
config = {
    'noise_injection': {
        'sensors': {
            'joint_position': {'type': 'gaussian', 'sigma': 0.01},
            'camera': {'type': 'salt_pepper', 'salt_prob': 0.02}
        }
    }
}

env = gym.make("FrankaPanda-v1")  # Example
env = NoiseInjectionWrapper(env, config=config)

# Train with SAC (off-policy, good for manipulation)
from stable_baselines3 import SAC
model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### Use Case 3: Mobile Robot Navigation

Train with LiDAR noise and wheel slip:

```python
config = {
    'noise_injection': {
        'sensors': {
            'lidar': {'type': 'gaussian', 'sigma': 0.03},  # 3cm noise
            'odometry': {'type': 'ornstein_uhlenbeck', 'sigma': 0.1}
        }
    }
}

env = gym.make("TurtleBot-v1")  # Example
env = NoiseInjectionWrapper(env, config=config)
```

## Advanced Features

### Particle Filter for Localization

```python
from stochastic_robotics.filters import ParticleFilter

def motion_model(state, control, rng):
    # Your motion model
    return next_state

def observation_model(state, observation):
    # Your sensor model
    return likelihood

pf = ParticleFilter(
    n_particles=1000,
    state_dim=3,  # x, y, theta
    motion_model=motion_model,
    observation_model=observation_model
)

# Update filter
for control, observation in zip(controls, observations):
    state_estimate, covariance = pf.step(control, observation)
```

### Custom Noise Models

Create your own noise model:

```python
from stochastic_robotics.noise_models import BaseNoise
import numpy as np

class MyCustomNoise(BaseNoise):
    def __init__(self, param1, param2, seed=None):
        super().__init__(seed)
        self.param1 = param1
        self.param2 = param2
    
    def apply(self, signal):
        # Implement your noise function
        noise = self.rng.custom_distribution(signal.shape)
        return signal + noise
    
    def reset(self):
        # Reset any internal state
        pass
```

## Troubleshooting

### Issue: Out of Memory During Training

**Solution**: Reduce number of parallel environments or particle count:
```bash
python train_robust_policy.py --n-envs 4  # Instead of 8
```

### Issue: Training is Too Slow

**Solution**: 
1. Use GPU if available: `--device cuda`
2. Reduce evaluation frequency: `--eval-freq 100000`
3. Use fewer particles in filters

### Issue: Policy Not Robust to Noise

**Solution**:
1. Increase noise range in domain randomization
2. Train for more timesteps
3. Use curriculum learning (start with low noise)

## Next Steps

- Read the [API Documentation](docs/api_reference.md)
- Check out [Example Experiments](experiments/)
- Join our [Discussions](https://github.com/yassine-cherni/stochastic-robotics-framework/discussions)
- Read the [Research Paper](docs/paper.pdf)

## Getting Help

- 📖 Documentation: [docs/](docs/)
- 💬 Discussions: GitHub Discussions
- 🐛 Bug Reports: GitHub Issues
- 📧 Email: cherniyassine@gomyrobot.com

Happy robot learning! 🤖
