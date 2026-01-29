"""
Base noise model interface for stochastic simulation.
"""
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
import numpy as np


class BaseNoise(ABC):
    """Abstract base class for all noise models."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize noise model.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        
    @abstractmethod
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply noise to a signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Noisy signal
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state of the noise model."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration of the noise model.
        
        Returns:
            Dictionary containing noise model parameters
        """
        return {
            "type": self.__class__.__name__,
            "seed": self.seed
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseNoise':
        """
        Create noise model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Instantiated noise model
        """
        raise NotImplementedError


class NoiseScheduler:
    """Scheduler for dynamically adjusting noise parameters during training."""
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        schedule_type: str = "linear",
        total_steps: int = 1000000
    ):
        """
        Initialize noise scheduler.
        
        Args:
            initial_value: Starting noise level
            final_value: Final noise level
            schedule_type: Type of schedule ('linear', 'exponential', 'constant')
            total_steps: Total number of training steps
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self) -> float:
        """
        Get current noise value and increment step.
        
        Returns:
            Current noise level
        """
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.schedule_type == "linear":
            value = self.initial_value + (self.final_value - self.initial_value) * progress
        elif self.schedule_type == "exponential":
            value = self.initial_value * (self.final_value / self.initial_value) ** progress
        elif self.schedule_type == "constant":
            value = self.initial_value
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        self.current_step += 1
        return value
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
