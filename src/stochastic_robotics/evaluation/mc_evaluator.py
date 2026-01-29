"""
Monte Carlo evaluation tools for policy robustness assessment.
"""
from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import pandas as pd
from scipy import stats


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    noise_level: float
    returns: np.ndarray
    success_rates: np.ndarray
    episode_lengths: np.ndarray
    mean_return: float
    std_return: float
    confidence_interval: Tuple[float, float]
    percentiles: Dict[int, float]
    success_rate: float


class MCEvaluator:
    """
    Monte Carlo evaluator for policy robustness assessment.
    
    Runs multiple episodes under different noise conditions and computes
    statistical performance metrics.
    """
    
    def __init__(
        self,
        env,
        policy: Callable,
        confidence_level: float = 0.95
    ):
        """
        Initialize MC evaluator.
        
        Args:
            env: Gymnasium environment (with noise injection)
            policy: Policy function (observation -> action)
            confidence_level: Confidence level for intervals
        """
        self.env = env
        self.policy = policy
        self.confidence_level = confidence_level
        
    def evaluate_episodes(
        self,
        n_episodes: int,
        max_steps: Optional[int] = None,
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            render: Whether to render environment
            verbose: Print progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        returns = []
        episode_lengths = []
        success_flags = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # Get action from policy
                if hasattr(self.policy, 'predict'):
                    # Stable-Baselines3 style
                    action, _ = self.policy.predict(obs, deterministic=True)
                else:
                    # Raw policy function
                    action = self.policy(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_return += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if max_steps is not None and episode_length >= max_steps:
                    truncated = True
            
            returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Check for success (if info contains success flag)
            success = info.get('is_success', False) or (episode_return > 0)
            success_flags.append(success)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Return={episode_return:.2f}, "
                      f"Length={episode_length}")
        
        returns = np.array(returns)
        episode_lengths = np.array(episode_lengths)
        success_flags = np.array(success_flags)
        
        # Compute statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Confidence interval
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        ci_margin = z_score * std_return / np.sqrt(n_episodes)
        confidence_interval = (mean_return - ci_margin, mean_return + ci_margin)
        
        # Percentiles
        percentiles = {
            5: np.percentile(returns, 5),
            25: np.percentile(returns, 25),
            50: np.percentile(returns, 50),
            75: np.percentile(returns, 75),
            95: np.percentile(returns, 95)
        }
        
        return {
            'returns': returns,
            'episode_lengths': episode_lengths,
            'success_flags': success_flags,
            'mean_return': mean_return,
            'std_return': std_return,
            'confidence_interval': confidence_interval,
            'percentiles': percentiles,
            'success_rate': np.mean(success_flags),
            'n_episodes': n_episodes
        }
    
    def evaluate_robustness(
        self,
        noise_levels: List[float],
        n_episodes: int = 100,
        noise_parameter: str = 'sigma',
        max_steps: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[float, EvaluationResult]:
        """
        Evaluate policy robustness across different noise levels.
        
        Args:
            noise_levels: List of noise levels to test
            n_episodes: Number of episodes per noise level
            noise_parameter: Which noise parameter to vary
            max_steps: Maximum steps per episode
            verbose: Print progress
            
        Returns:
            Dictionary mapping noise levels to evaluation results
        """
        results = {}
        
        for noise_level in noise_levels:
            if verbose:
                print(f"\nEvaluating at noise level: {noise_level}")
            
            # Update noise level in environment
            self._set_noise_level(noise_level, noise_parameter)
            
            # Evaluate
            eval_result = self.evaluate_episodes(
                n_episodes=n_episodes,
                max_steps=max_steps,
                render=False,
                verbose=verbose
            )
            
            # Store as EvaluationResult
            results[noise_level] = EvaluationResult(
                noise_level=noise_level,
                returns=eval_result['returns'],
                success_rates=eval_result['success_flags'],
                episode_lengths=eval_result['episode_lengths'],
                mean_return=eval_result['mean_return'],
                std_return=eval_result['std_return'],
                confidence_interval=eval_result['confidence_interval'],
                percentiles=eval_result['percentiles'],
                success_rate=eval_result['success_rate']
            )
        
        return results
    
    def _set_noise_level(self, level: float, parameter: str):
        """Set noise level in environment."""
        # This assumes env is NoiseInjectionWrapper
        if hasattr(self.env, 'obs_noise_models'):
            for noise_model in self.env.obs_noise_models.values():
                if hasattr(noise_model, parameter):
                    setattr(noise_model, parameter, level)
        
        if hasattr(self.env, 'action_noise_model'):
            if self.env.action_noise_model and hasattr(self.env.action_noise_model, parameter):
                setattr(self.env.action_noise_model, parameter, level)
    
    def plot_robustness_curves(
        self,
        results: Dict[float, EvaluationResult],
        save_path: Optional[str] = None
    ):
        """
        Plot robustness curves showing performance vs noise level.
        
        Args:
            results: Results from evaluate_robustness
            save_path: Optional path to save figure
        """
        noise_levels = sorted(results.keys())
        mean_returns = [results[level].mean_return for level in noise_levels]
        std_returns = [results[level].std_return for level in noise_levels]
        ci_lower = [results[level].confidence_interval[0] for level in noise_levels]
        ci_upper = [results[level].confidence_interval[1] for level in noise_levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Mean return with confidence intervals
        ax1.plot(noise_levels, mean_returns, 'o-', linewidth=2, markersize=8, label='Mean Return')
        ax1.fill_between(noise_levels, ci_lower, ci_upper, alpha=0.3, label=f'{self.confidence_level*100:.0f}% CI')
        ax1.set_xlabel('Noise Level (σ)', fontsize=12)
        ax1.set_ylabel('Mean Return', fontsize=12)
        ax1.set_title('Policy Robustness: Return vs Noise Level', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: Success rate
        success_rates = [results[level].success_rate * 100 for level in noise_levels]
        ax2.plot(noise_levels, success_rates, 's-', linewidth=2, markersize=8, color='green', label='Success Rate')
        ax2.set_xlabel('Noise Level (σ)', fontsize=12)
        ax2.set_ylabel('Success Rate (%)', fontsize=12)
        ax2.set_title('Policy Robustness: Success Rate vs Noise Level', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distribution_comparison(
        self,
        results: Dict[float, EvaluationResult],
        noise_levels_to_plot: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot return distributions for different noise levels.
        
        Args:
            results: Results from evaluate_robustness
            noise_levels_to_plot: Specific noise levels to plot (default: all)
            save_path: Optional path to save figure
        """
        if noise_levels_to_plot is None:
            noise_levels_to_plot = sorted(results.keys())
        
        fig, axes = plt.subplots(1, len(noise_levels_to_plot), 
                                figsize=(5*len(noise_levels_to_plot), 4),
                                sharey=True)
        
        if len(noise_levels_to_plot) == 1:
            axes = [axes]
        
        for idx, noise_level in enumerate(noise_levels_to_plot):
            result = results[noise_level]
            
            axes[idx].hist(result.returns, bins=20, alpha=0.7, edgecolor='black')
            axes[idx].axvline(result.mean_return, color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {result.mean_return:.2f}')
            axes[idx].axvline(result.percentiles[5], color='orange', linestyle=':', 
                            linewidth=2, label=f'5th %ile: {result.percentiles[5]:.2f}')
            axes[idx].set_xlabel('Return', fontsize=11)
            if idx == 0:
                axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'σ = {noise_level}', fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Return Distributions Across Noise Levels', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_statistics(self, results: Dict[float, EvaluationResult]):
        """
        Print comprehensive statistics table.
        
        Args:
            results: Results from evaluate_robustness
        """
        data = []
        for noise_level in sorted(results.keys()):
            result = results[noise_level]
            data.append({
                'Noise Level': f'{noise_level:.3f}',
                'Mean Return': f'{result.mean_return:.2f}',
                'Std Dev': f'{result.std_return:.2f}',
                '95% CI': f'[{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]',
                '5th %ile': f'{result.percentiles[5]:.2f}',
                '95th %ile': f'{result.percentiles[95]:.2f}',
                'Success Rate': f'{result.success_rate*100:.1f}%'
            })
        
        df = pd.DataFrame(data)
        print("\n" + "="*100)
        print("ROBUSTNESS EVALUATION RESULTS")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
