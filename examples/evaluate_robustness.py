"""
Example script for evaluating a trained policy's robustness.

Usage:
    python evaluate_robustness.py --model path/to/model.zip --env-id Ant-v4
"""
import argparse
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from stochastic_robotics.wrappers import NoiseInjectionWrapper
from stochastic_robotics.evaluation import MCEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy robustness")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--env-id", type=str, default="Ant-v4", help="Environment ID")
    parser.add_argument("--config", type=str, default="../configs/noise_config.yaml",
                       help="Noise configuration file")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                       help="Algorithm used")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="Episodes per noise level")
    parser.add_argument("--noise-levels", nargs="+", type=float,
                       default=[0.0, 0.02, 0.05, 0.1, 0.2],
                       help="Noise levels to evaluate")
    parser.add_argument("--save-dir", type=str, default="./eval_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model}")
    if args.algo == "PPO":
        model = PPO.load(args.model)
    elif args.algo == "SAC":
        model = SAC.load(args.model)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # Create environment
    print(f"Creating environment: {args.env_id}")
    env = gym.make(args.env_id)
    env = NoiseInjectionWrapper(
        env,
        config=args.config,
        observation_noise=True,
        action_noise=True
    )
    
    # Create evaluator
    evaluator = MCEvaluator(env, model, confidence_level=0.95)
    
    # Run evaluation
    print(f"\nEvaluating robustness across {len(args.noise_levels)} noise levels...")
    print(f"Episodes per level: {args.n_episodes}")
    print(f"Noise levels: {args.noise_levels}")
    
    results = evaluator.evaluate_robustness(
        noise_levels=args.noise_levels,
        n_episodes=args.n_episodes,
        noise_parameter='sigma',
        verbose=True
    )
    
    # Print statistics
    evaluator.print_statistics(results)
    
    # Plot results
    print("\nGenerating plots...")
    curve_path = save_dir / "robustness_curves.png"
    evaluator.plot_robustness_curves(results, save_path=str(curve_path))
    print(f"Robustness curves saved to: {curve_path}")
    
    dist_path = save_dir / "return_distributions.png"
    evaluator.plot_distribution_comparison(results, save_path=str(dist_path))
    print(f"Distribution comparison saved to: {dist_path}")
    
    print("\nEvaluation complete!")
    env.close()


if __name__ == "__main__":
    main()
