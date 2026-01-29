"""
Example training script for robust policy learning with domain randomization.

This script demonstrates:
1. Setting up a MuJoCo environment with noise injection
2. Wrapping with domain randomization
3. Training with PPO or SAC
4. Periodic evaluation and logging
"""
import argparse
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from wrappers.noise_injection import NoiseInjectionWrapper, DomainRandomizationWrapper
from evaluation.mc_evaluator import MCEvaluator


def make_env(env_id: str, config_path: str, rank: int, use_domain_randomization: bool = True):
    """
    Create a single environment with noise injection.
    
    Args:
        env_id: Gymnasium environment ID
        config_path: Path to noise configuration YAML
        rank: Unique ID for the environment
        use_domain_randomization: Whether to use domain randomization
        
    Returns:
        Function that creates the environment
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        
        # Apply noise injection
        env = NoiseInjectionWrapper(
            env,
            config=config_path,
            observation_noise=True,
            action_noise=True,
            seed=rank
        )
        
        # Apply domain randomization
        if use_domain_randomization:
            env = DomainRandomizationWrapper(
                env,
                config=config_path,
                seed=rank
            )
        
        return env
    
    return _init


def train(args):
    """Main training function."""
    
    print("=" * 80)
    print(f"Training {args.algo} on {args.env_id}")
    print(f"Config: {args.config}")
    print(f"Domain Randomization: {args.domain_randomization}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Environments: {args.n_envs}")
    print("=" * 80)
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create vectorized training environment
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(args.env_id, args.config, i, args.domain_randomization)
            for i in range(args.n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(args.env_id, args.config, 0, args.domain_randomization)
        ])
    
    # Create evaluation environment (without domain randomization)
    eval_env = DummyVecEnv([
        make_env(args.env_id, args.config, 1000, use_domain_randomization=False)
    ])
    
    # Initialize algorithm
    if args.algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(save_dir / "tensorboard"),
            verbose=1,
            device=args.device
        )
    elif args.algo == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log=str(save_dir / "tensorboard"),
            verbose=1,
            device=args.device
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = save_dir / "final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\nTraining complete!")
    
    # Run robustness evaluation
    if args.eval_robustness:
        print("\n" + "=" * 80)
        print("Running robustness evaluation...")
        print("=" * 80)
        
        eval_single_env = gym.make(args.env_id)
        eval_single_env = NoiseInjectionWrapper(
            eval_single_env,
            config=args.config,
            observation_noise=True,
            action_noise=True
        )
        
        evaluator = MCEvaluator(eval_single_env, model, confidence_level=0.95)
        
        results = evaluator.evaluate_robustness(
            noise_levels=[0.0, 0.02, 0.05, 0.1, 0.2],
            n_episodes=100,
            noise_parameter='sigma',
            verbose=True
        )
        
        # Print statistics
        evaluator.print_statistics(results)
        
        # Plot and save
        plot_path = save_dir / "robustness_curves.png"
        evaluator.plot_robustness_curves(results, save_path=str(plot_path))
        print(f"\nRobustness plots saved to: {plot_path}")
        
        eval_single_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train robust robotic policies")
    
    # Environment
    parser.add_argument("--env-id", type=str, default="Ant-v4",
                      help="Gymnasium environment ID")
    parser.add_argument("--config", type=str, default="configs/noise_config.yaml",
                      help="Path to noise configuration file")
    
    # Algorithm
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                      help="RL algorithm to use")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                      help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                      help="Number of parallel environments")
    parser.add_argument("--domain-randomization", action="store_true",
                      help="Use domain randomization")
    
    # Logging and saving
    parser.add_argument("--experiment-name", type=str, default="stochastic_exp",
                      help="Experiment name")
    parser.add_argument("--save-dir", type=str, default="./results",
                      help="Directory to save results")
    parser.add_argument("--save-freq", type=int, default=100_000,
                      help="Save frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                      help="Evaluation frequency (in timesteps)")
    
    # Evaluation
    parser.add_argument("--eval-robustness", action="store_true",
                      help="Run robustness evaluation after training")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {args.device}")
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
