"""
Ornstein-Uhlenbeck noise process for time-correlated noise.
"""
from typing import Optional, Dict, Any
import numpy as np
from .base_noise import BaseNoise


class OrnsteinUhlenbeckNoise(BaseNoise):
    """
    Ornstein-Uhlenbeck process for modeling time-correlated noise.
    
    The OU process follows:
        dx = theta * (mu - x) * dt + sigma * dW
    
    Where:
        - theta: Mean reversion rate
        - mu: Long-term mean
        - sigma: Volatility
        - dW: Wiener process increment
    
    This is ideal for modeling actuator drift, sensor bias, and other
    time-correlated disturbances.
    """
    
    def __init__(
        self,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None,
        clip_range: Optional[tuple] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Ornstein-Uhlenbeck noise process.
        
        Args:
            theta: Mean reversion rate (higher = faster reversion)
            mu: Long-term mean
            sigma: Volatility (noise magnitude)
            dt: Time step
            x0: Initial state (if None, starts at mu)
            clip_range: Optional (min, max) to clip values
            seed: Random seed
        """
        super().__init__(seed)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.clip_range = clip_range
        self.state = None
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply OU noise to signal.
        
        Args:
            signal: Input signal array
            
        Returns:
            Signal with additive OU noise
        """
        if self.state is None:
            # Initialize state
            if self.x0 is not None:
                self.state = np.array(self.x0)
            else:
                self.state = np.full(signal.shape, self.mu)
        
        # Ensure state matches signal shape
        if self.state.shape != signal.shape:
            self.state = np.full(signal.shape, self.mu)
        
        # OU process update
        dx = self.theta * (self.mu - self.state) * self.dt
        dw = self.rng.normal(0, np.sqrt(self.dt), size=signal.shape)
        self.state = self.state + dx + self.sigma * dw
        
        # Apply noise
        noisy_signal = signal + self.state
        
        if self.clip_range is not None:
            noisy_signal = np.clip(noisy_signal, self.clip_range[0], self.clip_range[1])
        
        return noisy_signal
    
    def reset(self):
        """Reset process to initial state."""
        self.state = None
    
    def get_current_noise(self) -> Optional[np.ndarray]:
        """
        Get current noise state without applying to signal.
        
        Returns:
            Current noise value or None if not initialized
        """
        return self.state.copy() if self.state is not None else None
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        config = super().get_config()
        config.update({
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "dt": self.dt,
            "x0": self.x0.tolist() if self.x0 is not None else None,
            "clip_range": self.clip_range
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'OrnsteinUhlenbeckNoise':
        """Create from configuration dictionary."""
        x0 = config.get("x0")
        if x0 is not None:
            x0 = np.array(x0)
        
        return cls(
            theta=config.get("theta", 0.15),
            mu=config.get("mu", 0.0),
            sigma=config.get("sigma", 0.2),
            dt=config.get("dt", 1e-2),
            x0=x0,
            clip_range=config.get("clip_range"),
            seed=config.get("seed")
        )
    
    @property
    def correlation_time(self) -> float:
        """
        Get correlation time of the process.
        
        Returns:
            Correlation time (1/theta)
        """
        return 1.0 / self.theta
    
    @property
    def stationary_variance(self) -> float:
        """
        Get stationary variance of the process.
        
        Returns:
            Variance at steady state (sigma^2 / (2*theta))
        """
        return (self.sigma ** 2) / (2 * self.theta)
