"""
Gaussian (Normal) noise model implementation.
"""
from typing import Optional, Union, Dict, Any
import numpy as np
from .base_noise import BaseNoise


class GaussianNoise(BaseNoise):
    """
    Gaussian (Normal) white noise model.
    
    Models additive noise with normal distribution:
        y = x + N(mu, sigma^2)
    """
    
    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.1,
        clip_range: Optional[tuple] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Gaussian noise model.
        
        Args:
            mu: Mean of the Gaussian distribution
            sigma: Standard deviation
            clip_range: Optional (min, max) to clip noisy values
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.mu = mu
        self.sigma = sigma
        self.clip_range = clip_range
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian noise to signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Noisy signal
        """
        noise = self.rng.normal(self.mu, self.sigma, size=signal.shape)
        noisy_signal = signal + noise
        
        if self.clip_range is not None:
            noisy_signal = np.clip(noisy_signal, self.clip_range[0], self.clip_range[1])
        
        return noisy_signal
    
    def reset(self):
        """Reset internal state (no state for white noise)."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "mu": self.mu,
            "sigma": self.sigma,
            "clip_range": self.clip_range
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GaussianNoise':
        """Create from configuration dictionary."""
        return cls(
            mu=config.get("mu", 0.0),
            sigma=config.get("sigma", 0.1),
            clip_range=config.get("clip_range", None),
            seed=config.get("seed", None)
        )


class MultivariateGaussianNoise(BaseNoise):
    """
    Multivariate Gaussian noise with correlation structure.
    
    Useful for modeling correlated sensor noise.
    """
    
    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        clip_range: Optional[tuple] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize multivariate Gaussian noise.
        
        Args:
            mean: Mean vector (n_dims,)
            covariance: Covariance matrix (n_dims, n_dims)
            clip_range: Optional (min, max) to clip noisy values
            seed: Random seed
        """
        super().__init__(seed)
        self.mean = np.array(mean)
        self.covariance = np.array(covariance)
        self.clip_range = clip_range
        
        # Validate dimensions
        assert self.mean.ndim == 1, "Mean must be 1D array"
        assert self.covariance.shape == (len(self.mean), len(self.mean)), \
            "Covariance must be square matrix matching mean dimension"
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply multivariate Gaussian noise.
        
        Args:
            signal: Input signal (n_samples, n_dims) or (n_dims,)
            
        Returns:
            Noisy signal
        """
        original_shape = signal.shape
        
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        n_samples = signal.shape[0]
        noise = self.rng.multivariate_normal(
            self.mean, 
            self.covariance, 
            size=n_samples
        )
        
        noisy_signal = signal + noise
        
        if self.clip_range is not None:
            noisy_signal = np.clip(noisy_signal, self.clip_range[0], self.clip_range[1])
        
        return noisy_signal.reshape(original_shape)
    
    def reset(self):
        """Reset internal state."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "clip_range": self.clip_range
        })
        return config
