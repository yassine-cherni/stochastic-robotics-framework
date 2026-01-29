"""
Salt-and-pepper (impulse) noise for image/sensor data.
"""
from typing import Optional, Dict, Any
import numpy as np
from .base_noise import BaseNoise


class SaltPepperNoise(BaseNoise):
    """
    Salt-and-pepper impulse noise model.
    
    Randomly replaces pixels/values with min (pepper) or max (salt) values.
    Common in image sensors, communication channels, and sensor dropouts.
    """
    
    def __init__(
        self,
        salt_prob: float = 0.01,
        pepper_prob: float = 0.01,
        salt_value: Optional[float] = None,
        pepper_value: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize salt-and-pepper noise.
        
        Args:
            salt_prob: Probability of salt (max value) noise
            pepper_prob: Probability of pepper (min value) noise
            salt_value: Value to use for salt (if None, uses signal max)
            pepper_value: Value to use for pepper (if None, uses signal min)
            seed: Random seed
        """
        super().__init__(seed)
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.salt_value = salt_value
        self.pepper_value = pepper_value
        
        assert 0 <= salt_prob <= 1, "salt_prob must be in [0, 1]"
        assert 0 <= pepper_prob <= 1, "pepper_prob must be in [0, 1]"
        assert salt_prob + pepper_prob <= 1, "salt_prob + pepper_prob must be <= 1"
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply salt-and-pepper noise to signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Noisy signal
        """
        noisy_signal = signal.copy()
        
        # Determine salt and pepper values
        salt_val = self.salt_value if self.salt_value is not None else signal.max()
        pepper_val = self.pepper_value if self.pepper_value is not None else signal.min()
        
        # Generate random mask
        rand = self.rng.random(signal.shape)
        
        # Apply salt noise
        salt_mask = rand < self.salt_prob
        noisy_signal[salt_mask] = salt_val
        
        # Apply pepper noise
        pepper_mask = (rand >= self.salt_prob) & (rand < self.salt_prob + self.pepper_prob)
        noisy_signal[pepper_mask] = pepper_val
        
        return noisy_signal
    
    def reset(self):
        """Reset internal state (no state for impulse noise)."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "salt_prob": self.salt_prob,
            "pepper_prob": self.pepper_prob,
            "salt_value": self.salt_value,
            "pepper_value": self.pepper_value
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SaltPepperNoise':
        """Create from configuration dictionary."""
        return cls(
            salt_prob=config.get("salt_prob", 0.01),
            pepper_prob=config.get("pepper_prob", 0.01),
            salt_value=config.get("salt_value"),
            pepper_value=config.get("pepper_value"),
            seed=config.get("seed")
        )


class DropoutNoise(BaseNoise):
    """
    Dropout noise - randomly sets values to zero.
    
    Models sensor dropouts, communication failures, etc.
    """
    
    def __init__(
        self,
        dropout_prob: float = 0.1,
        dropout_value: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize dropout noise.
        
        Args:
            dropout_prob: Probability of dropout for each value
            dropout_value: Value to use for dropped elements
            seed: Random seed
        """
        super().__init__(seed)
        self.dropout_prob = dropout_prob
        self.dropout_value = dropout_value
        
        assert 0 <= dropout_prob <= 1, "dropout_prob must be in [0, 1]"
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply dropout noise to signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Signal with random dropouts
        """
        noisy_signal = signal.copy()
        
        # Generate dropout mask
        dropout_mask = self.rng.random(signal.shape) < self.dropout_prob
        noisy_signal[dropout_mask] = self.dropout_value
        
        return noisy_signal
    
    def reset(self):
        """Reset internal state."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "dropout_prob": self.dropout_prob,
            "dropout_value": self.dropout_value
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DropoutNoise':
        """Create from configuration dictionary."""
        return cls(
            dropout_prob=config.get("dropout_prob", 0.1),
            dropout_value=config.get("dropout_value", 0.0),
            seed=config.get("seed")
        )
