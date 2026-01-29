"""
Gymnasium wrappers for noise injection and domain randomization.
"""
from typing import Dict, Any, Optional, Union, List
import numpy as np
import gymnasium as gym
import yaml

from ..noise_models import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    SaltPepperNoise,
    DropoutNoise
)


class NoiseInjectionWrapper(gym.Wrapper):
    """
    Wrapper that applies configurable noise to observations and actions.
    
    Supports different noise models for different sensor modalities.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Union[str, Dict[str, Any]],
        observation_noise: bool = True,
        action_noise: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize noise injection wrapper.
        
        Args:
            env: Base Gymnasium environment
            config: Configuration dict or path to YAML file
            observation_noise: Enable observation noise
            action_noise: Enable action noise
            seed: Random seed
        """
        super().__init__(env)
        
        # Load configuration
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.observation_noise_enabled = observation_noise
        self.action_noise_enabled = action_noise
        self.rng = np.random.RandomState(seed)
        
        # Initialize noise models
        self._init_observation_noise()
        self._init_action_noise()
        
        self.step_count = 0
        
    def _init_observation_noise(self):
        """Initialize observation noise models from config."""
        self.obs_noise_models = {}
        
        if not self.observation_noise_enabled:
            return
        
        noise_config = self.config.get('noise_injection', {}).get('sensors', {})
        
        for sensor_name, sensor_config in noise_config.items():
            noise_type = sensor_config.get('type', 'gaussian')
            
            if noise_type == 'gaussian':
                self.obs_noise_models[sensor_name] = GaussianNoise(
                    mu=sensor_config.get('mu', 0.0),
                    sigma=sensor_config.get('sigma', 0.1),
                    seed=self.rng.randint(0, 2**32 - 1)
                )
            elif noise_type == 'salt_pepper':
                self.obs_noise_models[sensor_name] = SaltPepperNoise(
                    salt_prob=sensor_config.get('salt_prob', 0.01),
                    pepper_prob=sensor_config.get('pepper_prob', 0.01),
                    seed=self.rng.randint(0, 2**32 - 1)
                )
            elif noise_type == 'dropout':
                self.obs_noise_models[sensor_name] = DropoutNoise(
                    dropout_prob=sensor_config.get('dropout_prob', 0.1),
                    seed=self.rng.randint(0, 2**32 - 1)
                )
    
    def _init_action_noise(self):
        """Initialize action noise models from config."""
        self.action_noise_model = None
        
        if not self.action_noise_enabled:
            return
        
        noise_config = self.config.get('noise_injection', {}).get('actuators', {})
        motor_config = noise_config.get('motor_commands', {})
        
        noise_type = motor_config.get('type', 'ornstein_uhlenbeck')
        
        if noise_type == 'ornstein_uhlenbeck':
            self.action_noise_model = OrnsteinUhlenbeckNoise(
                theta=motor_config.get('theta', 0.15),
                mu=motor_config.get('mu', 0.0),
                sigma=motor_config.get('sigma', 0.2),
                dt=motor_config.get('dt', 1e-2),
                seed=self.rng.randint(0, 2**32 - 1)
            )
        elif noise_type == 'gaussian':
            self.action_noise_model = GaussianNoise(
                mu=motor_config.get('mu', 0.0),
                sigma=motor_config.get('sigma', 0.1),
                seed=self.rng.randint(0, 2**32 - 1)
            )
    
    def _apply_observation_noise(self, observation: np.ndarray) -> np.ndarray:
        """Apply noise to observation."""
        if not self.observation_noise_enabled or not self.obs_noise_models:
            return observation
        
        # For dict observations
        if isinstance(observation, dict):
            noisy_obs = {}
            for key, value in observation.items():
                if key in self.obs_noise_models:
                    noisy_obs[key] = self.obs_noise_models[key].apply(value)
                else:
                    noisy_obs[key] = value
            return noisy_obs
        
        # For array observations - apply first noise model
        if self.obs_noise_models:
            noise_model = list(self.obs_noise_models.values())[0]
            return noise_model.apply(observation)
        
        return observation
    
    def _apply_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Apply noise to action."""
        if not self.action_noise_enabled or self.action_noise_model is None:
            return action
        
        return self.action_noise_model.apply(action)
    
    def reset(self, **kwargs):
        """Reset environment and noise models."""
        observation, info = self.env.reset(**kwargs)
        
        # Reset noise models
        for noise_model in self.obs_noise_models.values():
            noise_model.reset()
        
        if self.action_noise_model is not None:
            self.action_noise_model.reset()
        
        self.step_count = 0
        
        # Apply observation noise
        observation = self._apply_observation_noise(observation)
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Step environment with noisy action and return noisy observation."""
        # Apply action noise
        noisy_action = self._apply_action_noise(action)
        
        # Clip to action space
        if isinstance(self.env.action_space, gym.spaces.Box):
            noisy_action = np.clip(
                noisy_action,
                self.env.action_space.low,
                self.env.action_space.high
            )
        
        # Step environment
        observation, reward, terminated, truncated, info = self.env.step(noisy_action)
        
        # Apply observation noise
        observation = self._apply_observation_noise(observation)
        
        self.step_count += 1
        
        return observation, reward, terminated, truncated, info


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Wrapper for domain randomization during training.
    
    Randomizes noise parameters and physical properties at episode reset.
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: Union[str, Dict[str, Any]],
        seed: Optional[int] = None
    ):
        """
        Initialize domain randomization wrapper.
        
        Args:
            env: Base environment (should be NoiseInjectionWrapper)
            config: Configuration dict or YAML path
            seed: Random seed
        """
        super().__init__(env)
        
        # Load configuration
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.rng = np.random.RandomState(seed)
        
    def _randomize_noise_parameters(self):
        """Randomize noise parameters for the wrapped NoiseInjectionWrapper."""
        if not isinstance(self.env, NoiseInjectionWrapper):
            return
        
        noise_config = self.config.get('noise_injection', {}).get('sensors', {})
        
        for sensor_name, sensor_config in noise_config.items():
            if sensor_name not in self.env.obs_noise_models:
                continue
            
            noise_model = self.env.obs_noise_models[sensor_name]
            
            # Randomize sigma for Gaussian noise
            if isinstance(noise_model, GaussianNoise):
                sigma_range = sensor_config.get('sigma_range', [0.0, 0.1])
                noise_model.sigma = self.rng.uniform(*sigma_range)
            
            # Randomize probabilities for salt-pepper
            elif isinstance(noise_model, SaltPepperNoise):
                sp_range = sensor_config.get('prob_range', [0.0, 0.05])
                total_prob = self.rng.uniform(*sp_range)
                noise_model.salt_prob = total_prob / 2
                noise_model.pepper_prob = total_prob / 2
        
        # Randomize action noise
        if self.env.action_noise_model is not None:
            actuator_config = self.config.get('noise_injection', {}).get('actuators', {})
            motor_config = actuator_config.get('motor_commands', {})
            
            if isinstance(self.env.action_noise_model, OrnsteinUhlenbeckNoise):
                sigma_range = motor_config.get('sigma_range', [0.0, 0.2])
                self.env.action_noise_model.sigma = self.rng.uniform(*sigma_range)
    
    def _randomize_physics(self):
        """Randomize physics parameters if supported by environment."""
        if not hasattr(self.env.unwrapped, 'model'):
            return
        
        physics_config = self.config.get('physics_randomization', {})
        if not physics_config.get('enabled', False):
            return
        
        # This is environment-specific and would need to be customized
        # Example for MuJoCo environments
        dynamics = physics_config.get('dynamics', {})
        
        # Mass randomization
        if hasattr(self.env.unwrapped.model, 'body_mass'):
            mass_range = dynamics.get('mass_range', [0.8, 1.2])
            original_masses = self.env.unwrapped.model.body_mass.copy()
            multiplier = self.rng.uniform(*mass_range)
            self.env.unwrapped.model.body_mass[:] = original_masses * multiplier
    
    def reset(self, **kwargs):
        """Reset with randomized parameters."""
        # Randomize parameters
        self._randomize_noise_parameters()
        self._randomize_physics()
        
        # Reset environment
        return self.env.reset(**kwargs)
