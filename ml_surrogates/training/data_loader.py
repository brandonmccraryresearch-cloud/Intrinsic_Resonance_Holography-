"""
Data Loader for IRH Training Data

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Training data generation from numerical RG flow simulations

Provides datasets of RG trajectories for training ML surrogates:
- (initial_state, final_state) pairs
- Complete RG trajectories
- Fixed point examples
- Action functional values

Data augmentation strategies:
- Random initial conditions in coupling space
- Different target RG scales
- Various trajectory lengths
- Bootstrap sampling
"""

from typing import Dict, Generator, List, Optional, Tuple
import numpy as np

try:
    from ..engines.holographic_state import CouplingState, HolographicState
    from ..engines.resonance_engine import ResonanceEngine
except (ImportError, ValueError):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from engines.holographic_state import CouplingState, HolographicState
    from engines.resonance_engine import ResonanceEngine


class RGTrajectoryDataset:
    """
    Dataset of RG flow trajectories for training.
    
    Generates training data by running numerical RG integration
    and storing (input, target) pairs.
    
    Attributes:
        engine: ResonanceEngine for numerical integration
        coupling_ranges: Ranges for initial conditions
        scale_ranges: Ranges for RG scales
        num_samples: Number of samples in dataset
    """
    
    def __init__(
        self,
        coupling_ranges: Tuple[Tuple[float, float], ...] = (
            (5.0, 50.0),   # λ̃ range
            (5.0, 50.0),   # γ̃ range
            (5.0, 50.0),   # μ̃ range
        ),
        scale_range: Tuple[float, float] = (0.1, 1.0),
        num_samples: int = 1000,
        num_steps_range: Tuple[int, int] = (10, 100),
        random_seed: Optional[int] = None
    ):
        """
        Initialize RG trajectory dataset.
        
        Args:
            coupling_ranges: Ranges for (λ̃, γ̃, μ̃) initial values
            scale_range: Range for initial/final RG scales
            num_samples: Number of trajectory samples
            num_steps_range: Range for number of RG steps
            random_seed: Random seed for reproducibility
        """
        self.coupling_ranges = coupling_ranges
        self.scale_range = scale_range
        self.num_samples = num_samples
        self.num_steps_range = num_steps_range
        
        # Initialize engine for numerical integration
        self.engine = ResonanceEngine(
            tolerance=1e-6,
            max_iterations=1000
        )
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate dataset
        self.trajectories = []
        self._generate_dataset()
    
    def _generate_dataset(self) -> None:
        """Generate complete dataset of RG trajectories."""
        print(f"Generating {self.num_samples} RG trajectory samples...")
        
        for i in range(self.num_samples):
            # Sample random initial conditions
            initial_state = self._sample_initial_state()
            
            # Sample target scale
            k_initial = initial_state.k
            k_final = np.random.uniform(
                self.scale_range[0],
                min(k_initial * 0.9, self.scale_range[1])
            )
            
            # Sample number of steps
            num_steps = np.random.randint(*self.num_steps_range)
            
            # Integrate RG flow
            try:
                trajectory = self.engine.integrate_rg_flow(
                    initial_state,
                    k_final=k_final,
                    num_steps=num_steps,
                    method='rk4'
                )
                
                # Store trajectory
                self.trajectories.append({
                    'initial_state': initial_state,
                    'final_state': trajectory.get_current_state(),
                    'trajectory': trajectory,
                    'num_steps': num_steps,
                    'k_initial': k_initial,
                    'k_final': k_final
                })
                
            except Exception as e:
                print(f"Warning: Failed to generate trajectory {i}: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{self.num_samples} trajectories")
        
        print(f"✓ Dataset generation complete: {len(self.trajectories)} trajectories")
    
    def _sample_initial_state(self) -> CouplingState:
        """Sample random initial coupling state."""
        lambda_init = np.random.uniform(*self.coupling_ranges[0])
        gamma_init = np.random.uniform(*self.coupling_ranges[1])
        mu_init = np.random.uniform(*self.coupling_ranges[2])
        k_init = np.random.uniform(*self.scale_range)
        
        return CouplingState(
            lambda_tilde=lambda_init,
            gamma_tilde=gamma_init,
            mu_tilde=mu_init,
            k=k_init
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get trajectory sample by index."""
        if idx >= len(self.trajectories):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        return self.trajectories[idx]
    
    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> List[Dict]:
        """
        Get a batch of trajectory samples.
        
        Args:
            batch_size: Number of samples in batch
            shuffle: Whether to shuffle samples
            
        Returns:
            List of trajectory dictionaries
        """
        if shuffle:
            indices = np.random.choice(
                len(self.trajectories),
                size=min(batch_size, len(self.trajectories)),
                replace=False
            )
        else:
            indices = np.arange(min(batch_size, len(self.trajectories)))
        
        return [self.trajectories[i] for i in indices]
    
    def iterate_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Generator[List[Dict], None, None]:
        """
        Iterate over dataset in batches.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle before iteration
            
        Yields:
            Batches of trajectory dictionaries
        """
        indices = np.arange(len(self.trajectories))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield [self.trajectories[idx] for idx in batch_indices]
    
    def split_train_val(
        self,
        val_fraction: float = 0.2
    ) -> Tuple['RGTrajectoryDataset', 'RGTrajectoryDataset']:
        """
        Split dataset into training and validation sets.
        
        Args:
            val_fraction: Fraction of data for validation
            
        Returns:
            (train_dataset, val_dataset)
        """
        val_size = int(len(self.trajectories) * val_fraction)
        
        # Shuffle and split
        indices = np.random.permutation(len(self.trajectories))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Create new datasets
        train_dataset = RGTrajectoryDataset.__new__(RGTrajectoryDataset)
        val_dataset = RGTrajectoryDataset.__new__(RGTrajectoryDataset)
        
        # Copy attributes
        for dataset in [train_dataset, val_dataset]:
            dataset.coupling_ranges = self.coupling_ranges
            dataset.scale_range = self.scale_range
            dataset.num_steps_range = self.num_steps_range
            dataset.engine = self.engine
        
        # Assign trajectories
        train_dataset.trajectories = [self.trajectories[i] for i in train_indices]
        val_dataset.trajectories = [self.trajectories[i] for i in val_indices]
        
        train_dataset.num_samples = len(train_dataset.trajectories)
        val_dataset.num_samples = len(val_dataset.trajectories)
        
        return train_dataset, val_dataset


class FixedPointDataset:
    """
    Dataset of fixed point examples for classification training.
    
    Generates trajectories that converge to fixed points
    and negative examples that don't converge.
    """
    
    def __init__(
        self,
        num_positive: int = 500,
        num_negative: int = 500,
        random_seed: Optional[int] = None
    ):
        """
        Initialize fixed point dataset.
        
        Args:
            num_positive: Number of fixed point examples
            num_negative: Number of non-fixed point examples
            random_seed: Random seed for reproducibility
        """
        self.num_positive = num_positive
        self.num_negative = num_negative
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.engine = ResonanceEngine(tolerance=1e-6)
        
        self.examples = []
        self._generate_dataset()
    
    def _generate_dataset(self) -> None:
        """Generate fixed point classification dataset."""
        print(f"Generating fixed point dataset ({self.num_positive} positive, "
              f"{self.num_negative} negative)...")
        
        # Generate positive examples (converge to FP)
        for i in range(self.num_positive):
            # Start near expected fixed point region
            initial = CouplingState(
                lambda_tilde=np.random.uniform(40, 60),
                gamma_tilde=np.random.uniform(90, 110),
                mu_tilde=np.random.uniform(140, 170),
                k=np.random.uniform(0.5, 1.0)
            )
            
            # Flow to IR
            trajectory = self.engine.integrate_rg_flow(
                initial,
                k_final=0.01,
                num_steps=100,
                method='rk4'
            )
            
            # Check if converged
            is_fp = trajectory.check_fixed_point(tolerance=1e-4)
            
            self.examples.append({
                'trajectory': trajectory,
                'is_fixed_point': is_fp,
                'label': 1 if is_fp else 0
            })
        
        # Generate negative examples (don't converge)
        for i in range(self.num_negative):
            # Start far from fixed point
            initial = CouplingState(
                lambda_tilde=np.random.uniform(5, 20),
                gamma_tilde=np.random.uniform(5, 30),
                mu_tilde=np.random.uniform(5, 40),
                k=np.random.uniform(0.5, 1.0)
            )
            
            # Short flow
            trajectory = self.engine.integrate_rg_flow(
                initial,
                k_final=0.3,
                num_steps=20,
                method='euler'
            )
            
            self.examples.append({
                'trajectory': trajectory,
                'is_fixed_point': False,
                'label': 0
            })
        
        print(f"✓ Fixed point dataset complete: {len(self.examples)} examples")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        return self.examples[idx]
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[Dict]:
        """Get a batch of examples."""
        if shuffle:
            indices = np.random.choice(len(self.examples), size=batch_size, replace=False)
        else:
            indices = np.arange(batch_size)
        
        return [self.examples[i] for i in indices]


# Example usage and validation
if __name__ == "__main__":
    print("Testing RGTrajectoryDataset...")
    
    # Test 1: Basic dataset creation
    print("\n1. Creating small dataset...")
    dataset = RGTrajectoryDataset(
        num_samples=10,
        random_seed=42
    )
    
    print(f"  ✓ Dataset size: {len(dataset)}")
    
    # Test 2: Get single sample
    print("\n2. Getting sample...")
    sample = dataset[0]
    print(f"  ✓ Initial state: {sample['initial_state']}")
    print(f"  ✓ Final state: {sample['final_state']}")
    print(f"  ✓ Trajectory length: {sample['trajectory'].get_trajectory_length()}")
    
    # Test 3: Get batch
    print("\n3. Getting batch...")
    batch = dataset.get_batch(batch_size=5, shuffle=True)
    print(f"  ✓ Batch size: {len(batch)}")
    
    # Test 4: Iterate batches
    print("\n4. Iterating batches...")
    batch_count = 0
    for batch in dataset.iterate_batches(batch_size=3):
        batch_count += 1
    print(f"  ✓ Number of batches: {batch_count}")
    
    # Test 5: Train/val split
    print("\n5. Train/val split...")
    train_dataset, val_dataset = dataset.split_train_val(val_fraction=0.3)
    print(f"  ✓ Train size: {len(train_dataset)}")
    print(f"  ✓ Val size: {len(val_dataset)}")
    assert len(train_dataset) + len(val_dataset) == len(dataset)
    
    # Test 6: Fixed point dataset
    print("\n6. Creating fixed point dataset...")
    fp_dataset = FixedPointDataset(
        num_positive=5,
        num_negative=5,
        random_seed=42
    )
    print(f"  ✓ FP dataset size: {len(fp_dataset)}")
    
    fp_sample = fp_dataset[0]
    print(f"  ✓ Sample label: {fp_sample['label']}")
    print(f"  ✓ Is fixed point: {fp_sample['is_fixed_point']}")
    
    print("\n✅ All data loader tests passed!")
