"""
Visualization Utilities for IRH ML Surrogate

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Visualization tools for training, evaluation, and analysis

Provides plotting and visualization functions:
1. Training curves (loss, learning rate over epochs)
2. RG trajectory visualization (coupling evolution)
3. Comparison plots (ML vs numerical RG)
4. Speedup charts
5. Evaluation metric dashboards

Uses matplotlib for plotting with optional interactive backends.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    from ..engines import CouplingState, HolographicState
    from ..training import ModelEvaluator
except (ImportError, ValueError):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from engines import CouplingState, HolographicState
    from training import ModelEvaluator


class TrainingVisualizer:
    """
    Visualize training progress and metrics.
    
    Creates plots for:
    - Training and validation loss curves
    - Learning rate schedule
    - Loss components (coupling, FP, action)
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_training_curves(
        self,
        history: Dict[str, List],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Training history dict with 'train_loss', 'val_loss'
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate schedule
        if 'learning_rate' in history:
            ax2.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_loss_components(
        self,
        loss_history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot individual loss components over time.
        
        Args:
            loss_history: Dict with 'coupling', 'fixed_point', 'action', 'total'
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(loss_history['total']) + 1)
        
        components = ['coupling', 'fixed_point', 'action', 'total']
        colors = ['blue', 'orange', 'green', 'red']
        styles = ['-', '--', '-.', '-']
        
        for component, color, style in zip(components, colors, styles):
            if component in loss_history:
                linewidth = 3 if component == 'total' else 2
                ax.plot(epochs, loss_history[component], 
                       color=color, linestyle=style, linewidth=linewidth,
                       label=component.replace('_', ' ').title())
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Loss Components Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


class TrajectoryVisualizer:
    """
    Visualize RG flow trajectories.
    
    Creates plots for:
    - Coupling evolution (λ̃, γ̃, μ̃) vs RG scale
    - 3D trajectory plots in coupling space
    - Comparison: ML predictions vs numerical RG
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        """Initialize trajectory visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        self.figsize = figsize
    
    def plot_coupling_evolution(
        self,
        trajectory: HolographicState,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot coupling constants vs RG scale.
        
        Args:
            trajectory: HolographicState with RG trajectory
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Extract trajectory data
        states = []
        for i in range(trajectory.get_trajectory_length()):
            node = trajectory.graph_repr['nodes'][i]
            states.append({
                'lambda': node['lambda_tilde'],
                'gamma': node['gamma_tilde'],
                'mu': node['mu_tilde'],
                'k': node['k']
            })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        k_values = [s['k'] for s in states]
        
        # λ̃ evolution
        axes[0].plot(k_values, [s['lambda'] for s in states], 'b-', linewidth=2)
        axes[0].set_xlabel('RG Scale k', fontsize=12)
        axes[0].set_ylabel('λ̃', fontsize=12)
        axes[0].set_title('λ̃ Evolution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # γ̃ evolution
        axes[1].plot(k_values, [s['gamma'] for s in states], 'r-', linewidth=2)
        axes[1].set_xlabel('RG Scale k', fontsize=12)
        axes[1].set_ylabel('γ̃', fontsize=12)
        axes[1].set_title('γ̃ Evolution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # μ̃ evolution
        axes[2].plot(k_values, [s['mu'] for s in states], 'g-', linewidth=2)
        axes[2].set_xlabel('RG Scale k', fontsize=12)
        axes[2].set_ylabel('μ̃', fontsize=12)
        axes[2].set_title('μ̃ Evolution', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_trajectory_comparison(
        self,
        ml_trajectory: HolographicState,
        numerical_trajectory: HolographicState,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Compare ML vs numerical RG trajectories.
        
        Args:
            ml_trajectory: ML surrogate prediction
            numerical_trajectory: Numerical RG integration
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Extract ML trajectory
        ml_states = []
        for i in range(ml_trajectory.get_trajectory_length()):
            node = ml_trajectory.graph_repr['nodes'][i]
            ml_states.append({
                'lambda': node['lambda_tilde'],
                'gamma': node['gamma_tilde'],
                'mu': node['mu_tilde'],
                'k': node['k']
            })
        
        # Extract numerical trajectory
        num_states = []
        for i in range(numerical_trajectory.get_trajectory_length()):
            node = numerical_trajectory.graph_repr['nodes'][i]
            num_states.append({
                'lambda': node['lambda_tilde'],
                'gamma': node['gamma_tilde'],
                'mu': node['mu_tilde'],
                'k': node['k']
            })
        
        ml_k = [s['k'] for s in ml_states]
        num_k = [s['k'] for s in num_states]
        
        coupling_names = ['lambda', 'gamma', 'mu']
        titles = ['λ̃ Comparison', 'γ̃ Comparison', 'μ̃ Comparison']
        
        for idx, (name, title) in enumerate(zip(coupling_names, titles)):
            axes[idx].plot(ml_k, [s[name] for s in ml_states], 
                          'b-', linewidth=2, label='ML Surrogate')
            axes[idx].plot(num_k, [s[name] for s in num_states], 
                          'r--', linewidth=2, label='Numerical RG')
            axes[idx].set_xlabel('RG Scale k', fontsize=12)
            axes[idx].set_ylabel(name, fontsize=12)
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


class EvaluationVisualizer:
    """
    Visualize evaluation metrics and benchmarks.
    
    Creates plots for:
    - Metric comparison charts
    - Speedup bar charts
    - Error distribution histograms
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """Initialize evaluation visualizer."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        self.figsize = figsize
    
    def plot_evaluation_summary(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Create evaluation summary dashboard.
        
        Args:
            results: Results from ModelEvaluator.evaluate_all()
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Coupling prediction metrics
        ax1 = fig.add_subplot(gs[0, 0])
        coupling = results['coupling_predictions']
        metrics = ['MSE', 'MAE', 'R²']
        values = [coupling['mse'], coupling['mae'], coupling['r_squared']]
        colors = ['blue', 'orange', 'green']
        ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Coupling Prediction Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Fixed point classification
        ax2 = fig.add_subplot(gs[0, 1])
        fp = results['fixed_point_classification']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [fp['accuracy'], fp['precision'], fp['recall'], fp['f1_score']]
        ax2.bar(metrics, values, color='purple', alpha=0.7)
        ax2.set_title('Fixed Point Classification', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=10)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Speedup benchmark
        if 'speedup_benchmark' in results:
            ax3 = fig.add_subplot(gs[1, :])
            speedup = results['speedup_benchmark']
            
            categories = ['ML Time', 'Numerical Time']
            times = [
                speedup['ml_time_seconds'],
                speedup['numerical_time_seconds']
            ]
            colors = ['green', 'red']
            bars = ax3.bar(categories, times, color=colors, alpha=0.7)
            
            # Add speedup annotation
            ax3.text(0.5, max(times) * 0.9, 
                    f"Speedup: {speedup['speedup_factor']:.1f}x",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            ax3.set_title('Speedup Benchmark', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Time (seconds)', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Action prediction
        ax4 = fig.add_subplot(gs[2, 0])
        action = results['action_predictions']
        metrics = ['MSE', 'MAE', 'R²']
        values = [action['mse'], action['mae'], action['r_squared']]
        colors = ['red', 'orange', 'blue']
        ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Action Prediction Metrics', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Value', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Summary text
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        summary_text = f"""
        Evaluation Summary
        ──────────────────
        
        Coupling Predictions:
        • MSE: {coupling['mse']:.6f}
        • R²: {coupling['r_squared']:.6f}
        
        Fixed Point Classification:
        • Accuracy: {fp['accuracy']:.4f}
        • F1 Score: {fp['f1_score']:.4f}
        
        Action Predictions:
        • R²: {action['r_squared']:.6f}
        """
        
        if 'speedup_benchmark' in results:
            summary_text += f"\n        Speedup: {speedup['speedup_factor']:.1f}x"
        
        ax5.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', family='monospace')
        
        plt.suptitle('IRH ML Surrogate Evaluation Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


# Convenience function for quick plotting
def quick_plot_training(history: Dict, save_path: Optional[str] = None):
    """
    Quick plot of training history.
    
    Args:
        history: Training history dict
        save_path: Optional path to save figure
    """
    viz = TrainingVisualizer()
    viz.plot_training_curves(history, save_path=save_path)


def quick_plot_trajectory(trajectory: HolographicState, save_path: Optional[str] = None):
    """
    Quick plot of RG trajectory.
    
    Args:
        trajectory: HolographicState trajectory
        save_path: Optional path to save figure
    """
    viz = TrajectoryVisualizer()
    viz.plot_coupling_evolution(trajectory, save_path=save_path)


def quick_plot_evaluation(results: Dict, save_path: Optional[str] = None):
    """
    Quick plot of evaluation results.
    
    Args:
        results: Results from ModelEvaluator
        save_path: Optional path to save figure
    """
    viz = EvaluationVisualizer()
    viz.plot_evaluation_summary(results, save_path=save_path)


# Example usage
if __name__ == "__main__":
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
    else:
        print("Testing Visualization Tools...")
        
        # Test 1: Training curves
        print("\n1. Testing TrainingVisualizer...")
        history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
            'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
            'learning_rate': [0.01, 0.009, 0.008, 0.007, 0.006]
        }
        
        viz = TrainingVisualizer()
        print("  Creating training curves plot...")
        fig = viz.plot_training_curves(history, show=False)
        print("  ✓ Training curves plot created")
        plt.close(fig)
        
        # Test 2: Trajectory visualization
        print("\n2. Testing TrajectoryVisualizer...")
        from engines import CouplingState, HolographicState
        
        initial = CouplingState(10.0, 10.0, 10.0, 1.0)
        trajectory = HolographicState(initial)
        
        for i in range(5):
            new_state = CouplingState(
                10.0 + i,
                10.0 + i * 0.5,
                10.0 + i * 0.3,
                1.0 - i * 0.1
            )
            trajectory.add_rg_step(new_state)
        
        traj_viz = TrajectoryVisualizer()
        print("  Creating trajectory plot...")
        fig = traj_viz.plot_coupling_evolution(trajectory, show=False)
        print("  ✓ Trajectory plot created")
        plt.close(fig)
        
        # Test 3: Evaluation dashboard
        print("\n3. Testing EvaluationVisualizer...")
        results = {
            'coupling_predictions': {
                'mse': 0.001,
                'mae': 0.02,
                'mape': 1.5,
                'r_squared': 0.995
            },
            'fixed_point_classification': {
                'accuracy': 0.96,
                'precision': 0.94,
                'recall': 0.98,
                'f1_score': 0.96
            },
            'action_predictions': {
                'mse': 0.0001,
                'mae': 0.005,
                'r_squared': 0.999
            },
            'speedup_benchmark': {
                'speedup_factor': 150.5,
                'ml_time_seconds': 0.05,
                'numerical_time_seconds': 7.53
            }
        }
        
        eval_viz = EvaluationVisualizer()
        print("  Creating evaluation dashboard...")
        fig = eval_viz.plot_evaluation_summary(results, show=False)
        print("  ✓ Evaluation dashboard created")
        plt.close(fig)
        
        print("\n✅ All visualization tools tested successfully!")
        print("\nNote: Plots created but not displayed (show=False for testing)")
