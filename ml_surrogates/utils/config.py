"""
Configuration Management for IRH ML Surrogate

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Centralized configuration and hyperparameter management

Provides:
1. Configuration classes for all components
2. YAML/JSON loading and saving
3. Hyperparameter tracking
4. Experiment management
5. Default configurations

Enables reproducible experiments and easy parameter tuning.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. Install with: pip install pyyaml")


@dataclass
class ModelConfig:
    """Configuration for IRHTransformer model."""
    
    embed_dim: int = 128
    encoder_layers: int = 3
    encoder_heads: int = 4
    decoder_layers: int = 3
    decoder_heads: int = 8
    feedforward_dim: Optional[int] = None  # Defaults to 4 * embed_dim
    dropout: float = 0.1
    
    def __post_init__(self):
        """Set defaults after initialization."""
        if self.feedforward_dim is None:
            self.feedforward_dim = 4 * self.embed_dim
    
    @classmethod
    def small(cls) -> 'ModelConfig':
        """Small model configuration for testing."""
        return cls(embed_dim=32, encoder_layers=2, decoder_layers=2)
    
    @classmethod
    def medium(cls) -> 'ModelConfig':
        """Medium model configuration."""
        return cls(embed_dim=64, encoder_layers=2, decoder_layers=2)
    
    @classmethod
    def large(cls) -> 'ModelConfig':
        """Large model configuration for production."""
        return cls(embed_dim=128, encoder_layers=3, decoder_layers=3)
    
    @classmethod
    def xlarge(cls) -> 'ModelConfig':
        """Extra-large model configuration."""
        return cls(embed_dim=256, encoder_layers=4, decoder_layers=4)


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    
    num_train_samples: int = 1000
    num_val_samples: int = 200
    num_test_samples: int = 200
    coupling_range_min: float = 5.0
    coupling_range_max: float = 50.0
    scale_range_min: float = 0.1
    scale_range_max: float = 1.0
    num_steps_min: int = 10
    num_steps_max: int = 100
    random_seed: Optional[int] = 42
    
    @classmethod
    def small(cls) -> 'DataConfig':
        """Small dataset for testing."""
        return cls(
            num_train_samples=100,
            num_val_samples=20,
            num_test_samples=20
        )
    
    @classmethod
    def medium(cls) -> 'DataConfig':
        """Medium dataset."""
        return cls(
            num_train_samples=1000,
            num_val_samples=200,
            num_test_samples=200
        )
    
    @classmethod
    def large(cls) -> 'DataConfig':
        """Large dataset for production."""
        return cls(
            num_train_samples=10000,
            num_val_samples=2000,
            num_test_samples=2000
        )


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    lr_strategy: str = 'exponential'  # 'exponential', 'step', 'cosine', 'constant'
    lr_decay_rate: float = 0.95
    lr_step_epochs: List[int] = field(default_factory=lambda: [30, 60, 90])
    lr_step_factor: float = 0.5
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    verbose: bool = True
    use_jax: bool = False
    
    @classmethod
    def fast(cls) -> 'TrainingConfig':
        """Fast training for debugging."""
        return cls(num_epochs=10, batch_size=16)
    
    @classmethod
    def standard(cls) -> 'TrainingConfig':
        """Standard training configuration."""
        return cls(num_epochs=100, batch_size=32)
    
    @classmethod
    def thorough(cls) -> 'TrainingConfig':
        """Thorough training with early stopping."""
        return cls(
            num_epochs=200,
            batch_size=32,
            early_stopping_patience=20
        )


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    
    coupling_weight: float = 1.0
    fixed_point_weight: float = 0.5
    action_weight: float = 0.1
    trajectory_weight: float = 0.3
    use_trajectory_loss: bool = False
    
    @classmethod
    def coupling_focused(cls) -> 'LossConfig':
        """Focus on coupling prediction accuracy."""
        return cls(
            coupling_weight=2.0,
            fixed_point_weight=0.3,
            action_weight=0.05
        )
    
    @classmethod
    def balanced(cls) -> 'LossConfig':
        """Balanced weights."""
        return cls(
            coupling_weight=1.0,
            fixed_point_weight=0.5,
            action_weight=0.1
        )
    
    @classmethod
    def fixed_point_focused(cls) -> 'LossConfig':
        """Focus on fixed point classification."""
        return cls(
            coupling_weight=0.5,
            fixed_point_weight=2.0,
            action_weight=0.1
        )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    include_speedup: bool = True
    speedup_num_samples: int = 100
    speedup_num_steps: int = 100
    save_plots: bool = True
    plot_dir: str = 'plots'
    generate_report: bool = True
    report_path: str = 'evaluation_report.txt'


@dataclass
class IRHConfig:
    """
    Complete IRH ML Surrogate configuration.
    
    Combines all component configurations.
    """
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment_name: str = "irh_surrogate"
    description: str = ""
    
    @classmethod
    def default(cls) -> 'IRHConfig':
        """Default configuration."""
        return cls()
    
    @classmethod
    def small_test(cls) -> 'IRHConfig':
        """Small configuration for testing."""
        return cls(
            model=ModelConfig.small(),
            data=DataConfig.small(),
            training=TrainingConfig.fast(),
            experiment_name="test_run"
        )
    
    @classmethod
    def production(cls) -> 'IRHConfig':
        """Production configuration."""
        return cls(
            model=ModelConfig.large(),
            data=DataConfig.large(),
            training=TrainingConfig.thorough(),
            experiment_name="production_run"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'loss': asdict(self.loss),
            'evaluation': asdict(self.evaluation),
            'experiment_name': self.experiment_name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'IRHConfig':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            experiment_name=config_dict.get('experiment_name', 'irh_surrogate'),
            description=config_dict.get('description', '')
        )
    
    def save_json(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to JSON file
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, path: str) -> 'IRHConfig':
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            IRHConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to YAML file
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_yaml(cls, path: str) -> 'IRHConfig':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            IRHConfig instance
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=" * 60)
        print(f"IRH ML Surrogate Configuration: {self.experiment_name}")
        print("=" * 60)
        
        if self.description:
            print(f"\nDescription: {self.description}\n")
        
        print("Model Configuration:")
        print(f"  • Embedding dimension: {self.model.embed_dim}")
        print(f"  • Encoder layers: {self.model.encoder_layers}")
        print(f"  • Decoder layers: {self.model.decoder_layers}")
        print(f"  • Total parameters: ~{self._estimate_parameters():,}")
        
        print("\nData Configuration:")
        print(f"  • Training samples: {self.data.num_train_samples:,}")
        print(f"  • Validation samples: {self.data.num_val_samples:,}")
        print(f"  • Test samples: {self.data.num_test_samples:,}")
        
        print("\nTraining Configuration:")
        print(f"  • Epochs: {self.training.num_epochs}")
        print(f"  • Batch size: {self.training.batch_size}")
        print(f"  • Learning rate: {self.training.learning_rate}")
        print(f"  • LR strategy: {self.training.lr_strategy}")
        print(f"  • Early stopping: patience={self.training.early_stopping_patience}")
        
        print("\nLoss Configuration:")
        print(f"  • Coupling weight: {self.loss.coupling_weight}")
        print(f"  • Fixed point weight: {self.loss.fixed_point_weight}")
        print(f"  • Action weight: {self.loss.action_weight}")
        
        print("\n" + "=" * 60)
    
    def _estimate_parameters(self) -> int:
        """Estimate number of model parameters."""
        d = self.model.embed_dim
        # Rough estimate: embedding + attention + feedforward
        encoder_params = self.model.encoder_layers * (4 * d * d + 2 * d * self.model.feedforward_dim)
        decoder_params = self.model.decoder_layers * (4 * d * d + 2 * d * self.model.feedforward_dim)
        output_params = d * 10  # Prediction heads
        return encoder_params + decoder_params + output_params


class ExperimentTracker:
    """
    Track multiple experiments and their configurations.
    
    Maintains a registry of experiments for comparison.
    """
    
    def __init__(self, registry_path: str = "experiments.json"):
        """
        Initialize experiment tracker.
        
        Args:
            registry_path: Path to experiment registry file
        """
        self.registry_path = Path(registry_path)
        self.experiments = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load experiment registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self) -> None:
        """Save experiment registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def register_experiment(
        self,
        config: IRHConfig,
        results: Optional[Dict] = None
    ) -> None:
        """
        Register an experiment.
        
        Args:
            config: Experiment configuration
            results: Optional experiment results
        """
        experiment_data = {
            'config': config.to_dict(),
            'results': results or {}
        }
        
        self.experiments[config.experiment_name] = experiment_data
        self._save_registry()
    
    def get_experiment(self, name: str) -> Optional[Dict]:
        """
        Get experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment data or None
        """
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments."""
        return list(self.experiments.keys())
    
    def compare_experiments(self, names: List[str]) -> None:
        """
        Print comparison of experiments.
        
        Args:
            names: List of experiment names to compare
        """
        print("=" * 80)
        print("Experiment Comparison")
        print("=" * 80)
        
        for name in names:
            exp = self.experiments.get(name)
            if exp:
                print(f"\n{name}:")
                config = exp['config']
                print(f"  Model: {config['model']['embed_dim']}d, "
                      f"{config['model']['encoder_layers']} enc layers")
                print(f"  Data: {config['data']['num_train_samples']} train samples")
                print(f"  Training: {config['training']['num_epochs']} epochs, "
                      f"lr={config['training']['learning_rate']}")
                
                if exp.get('results'):
                    results = exp['results']
                    print(f"  Results:")
                    if 'best_val_loss' in results:
                        print(f"    • Best val loss: {results['best_val_loss']:.4f}")
                    if 'speedup' in results:
                        print(f"    • Speedup: {results['speedup']:.1f}x")
            else:
                print(f"\n{name}: NOT FOUND")
        
        print("\n" + "=" * 80)


# Example usage
if __name__ == "__main__":
    print("Testing Configuration Management...")
    
    # Test 1: Create default config
    print("\n1. Creating default configuration...")
    config = IRHConfig.default()
    config.print_summary()
    
    # Test 2: Create small test config
    print("\n2. Creating small test configuration...")
    test_config = IRHConfig.small_test()
    test_config.print_summary()
    
    # Test 3: Save and load JSON
    print("\n3. Testing JSON save/load...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "config.json")
        config.save_json(json_path)
        print(f"  ✓ Saved to {json_path}")
        
        loaded_config = IRHConfig.load_json(json_path)
        print(f"  ✓ Loaded from {json_path}")
        assert loaded_config.model.embed_dim == config.model.embed_dim
    
    # Test 4: Experiment tracker
    print("\n4. Testing ExperimentTracker...")
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "experiments.json")
        tracker = ExperimentTracker(registry_path)
        
        # Register experiments
        exp1 = IRHConfig.small_test()
        exp1.experiment_name = "test_exp_1"
        tracker.register_experiment(exp1, {'best_val_loss': 0.5})
        
        exp2 = IRHConfig.default()
        exp2.experiment_name = "test_exp_2"
        tracker.register_experiment(exp2, {'best_val_loss': 0.3, 'speedup': 150.0})
        
        print(f"  ✓ Registered {len(tracker.list_experiments())} experiments")
        
        # Compare
        tracker.compare_experiments(['test_exp_1', 'test_exp_2'])
    
    print("\n✅ All configuration management tools tested successfully!")
