"""
Quick demo script showing basic framework usage.

This script demonstrates:
1. Creating an environment with noise
2. Training a simple policy
3. Evaluating robustness
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from stochastic_robotics.wrappers import NoiseInjectionWrapper, DomainRandomizationWrapper
from stochastic_robotics.evaluation import MCEvaluator
from stochastic_robotics.noise_models import GaussianNoise, OrnsteinUhlenbeckNoise


def demo_noise_models():
    """Demonstrate different noise models."""
    print("=" * 80)
    print("DEMO 1: Noise Models")
    print("=" * 80)
    
    # Create a simple signal
    signal = np.linspace(0, 1, 100)
    
    # Gaussian noise
    gaussian = GaussianNoise(mu=0.0, sigma=0.1)
    noisy_gaussian = gaussian.apply(signal)
    print(f"Gaussian noise - Original mean: {signal.mean():.3f}, "
          f"Noisy mean: {noisy_gaussian.mean():.3f}")
    
    # Ornstein-Uhlenbeck noise (time-correlated)
    ou_noise = OrnsteinUhlenbeckNoise(theta=0.15, mu=0.0, sigma=0.2)
    noisy_ou = ou_noise.apply(signal)
    print(f"OU noise - Correlation time: {ou_noise.correlation_time:.3f}s")
    
    print("\n✓ Noise models demonstration complete!\n")


def demo_basic_training():
    """Demonstrate basic training with noise injection."""
    print("=" * 80)
    print("DEMO 2: Training with Noise Injection")
    print("=" * 80)
    
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create minimal noise config
    config = {
        'noise_injection': {
            'enabled': True,
            'sensors': {
                'observation': {
                    'type': 'gaussian',
                    'sigma': 0.05
                }
            },
            'actuators': {
                'motor_commands': {
                    'type': 'gaussian',
                    'sigma': 0.02
                }
            }
        }
    }
    
    # Wrap with noise
    env = NoiseInjectionWrapper(env, config=config)
    
    print("Training policy for 10,000 steps (quick demo)...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10_000)
    
    print("✓ Training complete!")
    
    # Test the policy
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"Test episode reward: {total_reward}")
    print("\n✓ Basic training demonstration complete!\n")
    
    env.close()
    return model


def demo_robustness_evaluation(model):
    """Demonstrate robustness evaluation."""
    print("=" * 80)
    print("DEMO 3: Robustness Evaluation")
    print("=" * 80)
    
    # Create evaluation environment
    env = gym.make("CartPole-v1")
    config = {
        'noise_injection': {
            'enabled': True,
            'sensors': {
                'observation': {
                    'type': 'gaussian',
                    'sigma': 0.05
                }
            },
            'actuators': {
                'motor_commands': {
                    'type': 'gaussian',
                    'sigma': 0.02
                }
            }
        }
    }
    env = NoiseInjectionWrapper(env, config=config)
    
    # Create evaluator
    evaluator = MCEvaluator(env, model, confidence_level=0.95)
    
    print("Evaluating policy at different noise levels...")
    results = evaluator.evaluate_robustness(
        noise_levels=[0.0, 0.05, 0.1],
        n_episodes=20,  # Small number for demo
        verbose=False
    )
    
    # Print results
    print("\nRobustness Results:")
    print("-" * 60)
    for noise_level, result in results.items():
        print(f"Noise σ={noise_level:.2f}: "
              f"Mean Return = {result.mean_return:.1f} ± {result.std_return:.1f}, "
              f"Success Rate = {result.success_rate*100:.0f}%")
    print("-" * 60)
    
    print("\n✓ Robustness evaluation complete!\n")
    
    env.close()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "STOCHASTIC ROBOTICS FRAMEWORK DEMO" + " " * 29 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Demo 1: Noise models
    demo_noise_models()
    
    # Demo 2: Training
    model = demo_basic_training()
    
    # Demo 3: Evaluation
    demo_robustness_evaluation(model)
    
    print("=" * 80)
    print("ALL DEMOS COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check out examples/train_robust_policy.py for full training")
    print("2. Customize configs/noise_config.yaml for your robot")
    print("3. Read the documentation in docs/")
    print("\nHappy robot learning! 🤖")


if __name__ == "__main__":
    main()
