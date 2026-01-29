"""
Particle Filter (Sequential Monte Carlo) implementation for state estimation.
"""
from typing import Callable, Optional, Tuple, Union
import numpy as np


class ParticleFilter:
    """
    Sequential Importance Resampling (SIR) Particle Filter.
    
    Implements Monte Carlo Localization for nonlinear, non-Gaussian systems.
    """
    
    def __init__(
        self,
        n_particles: int,
        state_dim: int,
        motion_model: Callable,
        observation_model: Callable,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None,
        resample_threshold: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize particle filter.
        
        Args:
            n_particles: Number of particles
            state_dim: Dimension of state space
            motion_model: Function(state, control, noise) -> next_state
            observation_model: Function(state, observation) -> likelihood
            initial_state: Initial state estimate (mean)
            initial_covariance: Initial uncertainty (covariance)
            resample_threshold: Resample when N_eff < threshold * N
            seed: Random seed
        """
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.resample_threshold = resample_threshold
        self.rng = np.random.RandomState(seed)
        
        # Initialize particles
        if initial_state is None:
            initial_state = np.zeros(state_dim)
        if initial_covariance is None:
            initial_covariance = np.eye(state_dim)
        
        self.particles = self.rng.multivariate_normal(
            initial_state,
            initial_covariance,
            size=n_particles
        )
        self.weights = np.ones(n_particles) / n_particles
        
        # Statistics
        self.n_eff_history = []
        self.resample_count = 0
        
    def predict(self, control: np.ndarray):
        """
        Prediction step: propagate particles through motion model.
        
        Args:
            control: Control input
        """
        for i in range(self.n_particles):
            self.particles[i] = self.motion_model(self.particles[i], control, self.rng)
    
    def update(self, observation: np.ndarray):
        """
        Update step: weight particles by observation likelihood.
        
        Args:
            observation: Sensor measurement
        """
        for i in range(self.n_particles):
            self.weights[i] = self.observation_model(self.particles[i], observation)
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # All particles have zero likelihood - reset uniformly
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def resample(self, method: str = "systematic"):
        """
        Resample particles based on weights.
        
        Args:
            method: Resampling method ('systematic', 'multinomial', 'residual')
        """
        if method == "systematic":
            indices = self._systematic_resample()
        elif method == "multinomial":
            indices = self._multinomial_resample()
        elif method == "residual":
            indices = self._residual_resample()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.resample_count += 1
    
    def _systematic_resample(self) -> np.ndarray:
        """Systematic resampling (low variance)."""
        positions = (np.arange(self.n_particles) + self.rng.random()) / self.n_particles
        cumulative_sum = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative_sum, positions)
        return indices
    
    def _multinomial_resample(self) -> np.ndarray:
        """Multinomial resampling."""
        return self.rng.choice(
            self.n_particles,
            size=self.n_particles,
            replace=True,
            p=self.weights
        )
    
    def _residual_resample(self) -> np.ndarray:
        """Residual resampling."""
        indices = []
        # Take integer part
        n_copies = np.floor(self.n_particles * self.weights).astype(int)
        for i in range(self.n_particles):
            indices.extend([i] * n_copies[i])
        
        # Resample remainder
        residual = self.weights - n_copies / self.n_particles
        residual /= np.sum(residual)
        n_remainder = self.n_particles - len(indices)
        indices.extend(self.rng.choice(self.n_particles, size=n_remainder, p=residual))
        
        return np.array(indices)
    
    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted mean and covariance of particles.
        
        Returns:
            mean: Weighted mean state
            covariance: Weighted covariance
        """
        mean = np.average(self.particles, weights=self.weights, axis=0)
        
        diff = self.particles - mean
        covariance = np.dot(self.weights * diff.T, diff)
        
        return mean, covariance
    
    def effective_sample_size(self) -> float:
        """
        Compute effective sample size.
        
        Returns:
            N_eff = 1 / sum(weights^2)
        """
        return 1.0 / np.sum(self.weights ** 2)
    
    def step(
        self,
        control: np.ndarray,
        observation: np.ndarray,
        resample_method: str = "systematic"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete filter step: predict, update, optionally resample.
        
        Args:
            control: Control input
            observation: Sensor measurement
            resample_method: Resampling method
            
        Returns:
            mean: State estimate
            covariance: Uncertainty estimate
        """
        # Predict
        self.predict(control)
        
        # Update
        self.update(observation)
        
        # Check if resampling needed
        n_eff = self.effective_sample_size()
        self.n_eff_history.append(n_eff)
        
        if n_eff < self.resample_threshold * self.n_particles:
            self.resample(method=resample_method)
        
        # Estimate
        return self.estimate()
    
    def get_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current particles and weights.
        
        Returns:
            particles: Array of particles (n_particles, state_dim)
            weights: Array of weights (n_particles,)
        """
        return self.particles.copy(), self.weights.copy()
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        """
        Reset particle filter to initial state.
        
        Args:
            initial_state: Initial state estimate
            initial_covariance: Initial uncertainty
        """
        if initial_state is None:
            initial_state = np.zeros(self.state_dim)
        if initial_covariance is None:
            initial_covariance = np.eye(self.state_dim)
        
        self.particles = self.rng.multivariate_normal(
            initial_state,
            initial_covariance,
            size=self.n_particles
        )
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.n_eff_history = []
        self.resample_count = 0


class AdaptiveParticleFilter(ParticleFilter):
    """
    Adaptive Monte Carlo Localization (AMCL).
    
    Dynamically adjusts particle count based on uncertainty.
    """
    
    def __init__(
        self,
        min_particles: int = 500,
        max_particles: int = 5000,
        kld_epsilon: float = 0.05,
        kld_z: float = 3.0,
        **kwargs
    ):
        """
        Initialize adaptive particle filter.
        
        Args:
            min_particles: Minimum number of particles
            max_particles: Maximum number of particles
            kld_epsilon: KLD sampling error
            kld_z: Standard normal quantile (e.g., 3.0 for 99.7%)
            **kwargs: Arguments for ParticleFilter
        """
        kwargs['n_particles'] = max_particles
        super().__init__(**kwargs)
        
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.kld_epsilon = kld_epsilon
        self.kld_z = kld_z
        
    def adapt_particles(self):
        """Adjust particle count based on KLD criterion."""
        # Compute required particles using KLD-sampling
        # This is a simplified version - full implementation requires binning
        n_eff = self.effective_sample_size()
        
        # Heuristic: adjust based on effective sample size
        target_particles = int(np.clip(
            2 * self.n_particles - n_eff,
            self.min_particles,
            self.max_particles
        ))
        
        if target_particles < self.n_particles:
            # Reduce particles
            indices = self.rng.choice(
                self.n_particles,
                size=target_particles,
                replace=False,
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = self.weights[indices]
            self.weights /= np.sum(self.weights)
            self.n_particles = target_particles
