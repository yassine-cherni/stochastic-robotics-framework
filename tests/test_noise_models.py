"""
Unit tests for noise models.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from stochastic_robotics.noise_models import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    SaltPepperNoise,
    DropoutNoise
)


class TestGaussianNoise:
    """Test Gaussian noise model."""
    
    def test_initialization(self):
        """Test basic initialization."""
        noise = GaussianNoise(mu=0.0, sigma=0.1, seed=42)
        assert noise.mu == 0.0
        assert noise.sigma == 0.1
        assert noise.seed == 42
    
    def test_apply_shape(self):
        """Test that noise preserves signal shape."""
        noise = GaussianNoise(sigma=0.1, seed=42)
        signal = np.ones((10, 5))
        noisy_signal = noise.apply(signal)
        assert noisy_signal.shape == signal.shape
    
    def test_noise_statistics(self):
        """Test that noise has correct statistics."""
        noise = GaussianNoise(mu=0.0, sigma=0.1, seed=42)
        signal = np.zeros(10000)
        noisy_signal = noise.apply(signal)
        
        noise_values = noisy_signal - signal
        assert abs(np.mean(noise_values) - 0.0) < 0.01  # Mean close to 0
        assert abs(np.std(noise_values) - 0.1) < 0.01   # Std close to 0.1
    
    def test_clipping(self):
        """Test value clipping."""
        noise = GaussianNoise(sigma=1.0, clip_range=(-0.5, 0.5), seed=42)
        signal = np.zeros(100)
        noisy_signal = noise.apply(signal)
        
        assert np.all(noisy_signal >= -0.5)
        assert np.all(noisy_signal <= 0.5)


class TestOrnsteinUhlenbeckNoise:
    """Test Ornstein-Uhlenbeck noise."""
    
    def test_initialization(self):
        """Test initialization."""
        noise = OrnsteinUhlenbeckNoise(theta=0.15, mu=0.0, sigma=0.2, seed=42)
        assert noise.theta == 0.15
        assert noise.mu == 0.0
        assert noise.sigma == 0.2
    
    def test_correlation_time(self):
        """Test correlation time property."""
        noise = OrnsteinUhlenbeckNoise(theta=0.15, seed=42)
        expected_corr_time = 1.0 / 0.15
        assert abs(noise.correlation_time - expected_corr_time) < 1e-6
    
    def test_stationary_variance(self):
        """Test stationary variance property."""
        theta, sigma = 0.15, 0.2
        noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=sigma, seed=42)
        expected_var = (sigma ** 2) / (2 * theta)
        assert abs(noise.stationary_variance - expected_var) < 1e-6
    
    def test_time_correlation(self):
        """Test that noise is time-correlated."""
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, dt=0.01, seed=42)
        signal = np.zeros(1000)
        
        noise_sequence = []
        for _ in range(1000):
            noisy_signal = noise.apply(signal[:1])
            noise_sequence.append(noisy_signal[0])
        
        noise_sequence = np.array(noise_sequence)
        
        # Check autocorrelation at lag 1
        autocorr = np.corrcoef(noise_sequence[:-1], noise_sequence[1:])[0, 1]
        assert autocorr > 0.5  # Should be significantly correlated
    
    def test_reset(self):
        """Test that reset clears state."""
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, seed=42)
        signal = np.zeros(10)
        
        # Apply noise
        noise.apply(signal)
        assert noise.state is not None
        
        # Reset
        noise.reset()
        assert noise.state is None


class TestSaltPepperNoise:
    """Test salt-and-pepper noise."""
    
    def test_initialization(self):
        """Test initialization."""
        noise = SaltPepperNoise(salt_prob=0.1, pepper_prob=0.1, seed=42)
        assert noise.salt_prob == 0.1
        assert noise.pepper_prob == 0.1
    
    def test_probability_validation(self):
        """Test that invalid probabilities raise errors."""
        with pytest.raises(AssertionError):
            SaltPepperNoise(salt_prob=0.6, pepper_prob=0.6)  # Sum > 1
        
        with pytest.raises(AssertionError):
            SaltPepperNoise(salt_prob=-0.1)  # Negative probability
    
    def test_apply_changes_values(self):
        """Test that noise changes some values."""
        noise = SaltPepperNoise(salt_prob=0.2, pepper_prob=0.2, seed=42)
        signal = 0.5 * np.ones(100)
        noisy_signal = noise.apply(signal)
        
        # Some values should be different
        assert not np.allclose(signal, noisy_signal)


class TestDropoutNoise:
    """Test dropout noise."""
    
    def test_initialization(self):
        """Test initialization."""
        noise = DropoutNoise(dropout_prob=0.2, seed=42)
        assert noise.dropout_prob == 0.2
        assert noise.dropout_value == 0.0
    
    def test_dropout_rate(self):
        """Test that dropout occurs at correct rate."""
        noise = DropoutNoise(dropout_prob=0.3, seed=42)
        signal = np.ones(10000)
        noisy_signal = noise.apply(signal)
        
        dropout_rate = np.sum(noisy_signal == 0.0) / len(signal)
        assert abs(dropout_rate - 0.3) < 0.02  # Within 2% of target


def test_noise_from_config():
    """Test creating noise from configuration."""
    config = {
        "type": "GaussianNoise",
        "mu": 0.5,
        "sigma": 0.2,
        "seed": 42
    }
    
    noise = GaussianNoise.from_config(config)
    assert noise.mu == 0.5
    assert noise.sigma == 0.2
    assert noise.seed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
