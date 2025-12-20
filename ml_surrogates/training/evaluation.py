"""
Evaluation Metrics for IRH Surrogate Models

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Performance validation and benchmarking

Metrics for assessing ML surrogate performance:
1. Trajectory Error: MSE on RG flow trajectories
2. Fixed Point Accuracy: Classification accuracy
3. Action R²: Coefficient of determination for action predictions
4. Speedup Benchmark: Wall-clock time comparison vs numerical RG
5. Generalization: Performance on unseen initial conditions

Provides comprehensive evaluation suite for model validation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import time
from pathlib import Path

try:
    from ..models import IRHTransformer
    from ..engines import CouplingState, HolographicState, ResonanceEngine
    from ..training import RGTrajectoryDataset
except (ImportError, ValueError):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import IRHTransformer
    from engines import CouplingState, HolographicState, ResonanceEngine
    from training import RGTrajectoryDataset


class TrajectoryErrorMetrics:
    """
    Metrics for RG trajectory prediction accuracy.
    
    Computes various error metrics between predicted and true trajectories:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - R² coefficient of determination
    """
    
    @staticmethod
    def mse(predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute MSE."""
        return np.mean((predicted - target) ** 2)
    
    @staticmethod
    def mae(predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute MAE."""
        return np.mean(np.abs(predicted - target))
    
    @staticmethod
    def mape(predicted: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
        """Compute MAPE (Mean Absolute Percentage Error)."""
        return np.mean(np.abs((target - predicted) / (np.abs(target) + epsilon))) * 100
    
    @staticmethod
    def r_squared(predicted: np.ndarray, target: np.ndarray) -> float:
        """Compute R² coefficient."""
        ss_res = np.sum((target - predicted) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def compute_all(
        predicted: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all trajectory error metrics.
        
        Args:
            predicted: Predicted values (batch, 3) for (λ̃, γ̃, μ̃)
            target: Target values (batch, 3)
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mse': TrajectoryErrorMetrics.mse(predicted, target),
            'mae': TrajectoryErrorMetrics.mae(predicted, target),
            'mape': TrajectoryErrorMetrics.mape(predicted, target),
            'r_squared': TrajectoryErrorMetrics.r_squared(predicted, target)
        }


class FixedPointMetrics:
    """
    Metrics for fixed point classification.
    
    Evaluates binary classification performance:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
    """
    
    @staticmethod
    def accuracy(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Compute classification accuracy."""
        pred_labels = (predicted > threshold).astype(int)
        target_labels = target.astype(int)
        return np.mean(pred_labels == target_labels)
    
    @staticmethod
    def precision(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Compute precision."""
        pred_labels = (predicted > threshold).astype(int)
        target_labels = target.astype(int)
        
        true_positives = np.sum((pred_labels == 1) & (target_labels == 1))
        predicted_positives = np.sum(pred_labels == 1)
        
        if predicted_positives == 0:
            return 0.0
        
        return true_positives / predicted_positives
    
    @staticmethod
    def recall(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Compute recall."""
        pred_labels = (predicted > threshold).astype(int)
        target_labels = target.astype(int)
        
        true_positives = np.sum((pred_labels == 1) & (target_labels == 1))
        actual_positives = np.sum(target_labels == 1)
        
        if actual_positives == 0:
            return 0.0
        
        return true_positives / actual_positives
    
    @staticmethod
    def f1_score(predicted: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Compute F1 score."""
        prec = FixedPointMetrics.precision(predicted, target, threshold)
        rec = FixedPointMetrics.recall(predicted, target, threshold)
        
        if prec + rec == 0:
            return 0.0
        
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def confusion_matrix(
        predicted: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            2x2 array [[TN, FP], [FN, TP]]
        """
        pred_labels = (predicted > threshold).astype(int)
        target_labels = target.astype(int)
        
        tn = np.sum((pred_labels == 0) & (target_labels == 0))
        fp = np.sum((pred_labels == 1) & (target_labels == 0))
        fn = np.sum((pred_labels == 0) & (target_labels == 1))
        tp = np.sum((pred_labels == 1) & (target_labels == 1))
        
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def compute_all(
        predicted: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute all fixed point classification metrics.
        
        Args:
            predicted: Predicted probabilities (batch,)
            target: Target labels 0/1 (batch,)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': FixedPointMetrics.accuracy(predicted, target, threshold),
            'precision': FixedPointMetrics.precision(predicted, target, threshold),
            'recall': FixedPointMetrics.recall(predicted, target, threshold),
            'f1_score': FixedPointMetrics.f1_score(predicted, target, threshold),
            'confusion_matrix': FixedPointMetrics.confusion_matrix(predicted, target, threshold).tolist()
        }


class SpeedupBenchmark:
    """
    Benchmark speedup of ML surrogate vs numerical RG integration.
    
    Measures wall-clock time for:
    - ML surrogate forward pass
    - Numerical RG integration
    - Computes speedup factor
    """
    
    def __init__(
        self,
        model: IRHTransformer,
        engine: ResonanceEngine
    ):
        """
        Initialize benchmark.
        
        Args:
            model: Trained ML surrogate
            engine: Numerical RG engine
        """
        self.model = model
        self.engine = engine
    
    def benchmark_ml_surrogate(
        self,
        initial_states: List[CouplingState],
        target_scale: float
    ) -> Tuple[float, List[CouplingState]]:
        """
        Benchmark ML surrogate predictions.
        
        Args:
            initial_states: List of initial states
            target_scale: Target RG scale
            
        Returns:
            (elapsed_time, predicted_states)
        """
        start = time.time()
        
        predicted_states = []
        for initial in initial_states:
            predicted = self.model.predict_final_state(initial, target_scale)
            predicted_states.append(predicted)
        
        elapsed = time.time() - start
        
        return elapsed, predicted_states
    
    def benchmark_numerical_rg(
        self,
        initial_states: List[CouplingState],
        target_scale: float,
        num_steps: int = 100
    ) -> Tuple[float, List[CouplingState]]:
        """
        Benchmark numerical RG integration.
        
        Args:
            initial_states: List of initial states
            target_scale: Target RG scale
            num_steps: Number of integration steps
            
        Returns:
            (elapsed_time, final_states)
        """
        start = time.time()
        
        final_states = []
        for initial in initial_states:
            trajectory = self.engine.integrate_rg_flow(
                initial,
                k_final=target_scale,
                num_steps=num_steps,
                method='rk4'
            )
            final_states.append(trajectory.get_current_state())
        
        elapsed = time.time() - start
        
        return elapsed, final_states
    
    def compute_speedup(
        self,
        num_samples: int = 100,
        coupling_range: Tuple[float, float] = (10.0, 50.0),
        scale_range: Tuple[float, float] = (0.5, 1.0),
        target_scale: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute speedup factor.
        
        Args:
            num_samples: Number of samples to test
            coupling_range: Range for initial coupling values
            scale_range: Range for initial RG scales
            target_scale: Target RG scale
            
        Returns:
            Dictionary with timing and speedup info
        """
        # Generate random initial states
        initial_states = []
        for _ in range(num_samples):
            lambda_init = np.random.uniform(*coupling_range)
            gamma_init = np.random.uniform(*coupling_range)
            mu_init = np.random.uniform(*coupling_range)
            k_init = np.random.uniform(*scale_range)
            
            initial_states.append(CouplingState(
                lambda_tilde=lambda_init,
                gamma_tilde=gamma_init,
                mu_tilde=mu_init,
                k=k_init
            ))
        
        # Benchmark ML surrogate
        ml_time, ml_predictions = self.benchmark_ml_surrogate(
            initial_states,
            target_scale
        )
        
        # Benchmark numerical RG
        numerical_time, numerical_predictions = self.benchmark_numerical_rg(
            initial_states,
            target_scale
        )
        
        # Compute speedup
        speedup = numerical_time / ml_time if ml_time > 0 else float('inf')
        
        # Compute accuracy
        ml_couplings = np.array([
            [s.lambda_tilde, s.gamma_tilde, s.mu_tilde]
            for s in ml_predictions
        ])
        
        numerical_couplings = np.array([
            [s.lambda_tilde, s.gamma_tilde, s.mu_tilde]
            for s in numerical_predictions
        ])
        
        accuracy_metrics = TrajectoryErrorMetrics.compute_all(
            ml_couplings,
            numerical_couplings
        )
        
        return {
            'num_samples': num_samples,
            'ml_time_seconds': ml_time,
            'numerical_time_seconds': numerical_time,
            'speedup_factor': speedup,
            'ml_time_per_sample_ms': (ml_time / num_samples) * 1000,
            'numerical_time_per_sample_ms': (numerical_time / num_samples) * 1000,
            'accuracy_metrics': accuracy_metrics
        }


class ModelEvaluator:
    """
    Complete model evaluation suite.
    
    Combines all metrics and provides comprehensive model assessment.
    """
    
    def __init__(
        self,
        model: IRHTransformer,
        test_dataset: RGTrajectoryDataset,
        engine: Optional[ResonanceEngine] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            test_dataset: Test dataset
            engine: Numerical RG engine for speedup benchmarking
        """
        self.model = model
        self.test_dataset = test_dataset
        self.engine = engine or ResonanceEngine()
        self.benchmark = SpeedupBenchmark(model, self.engine)
    
    def evaluate_coupling_predictions(self) -> Dict[str, float]:
        """
        Evaluate coupling constant predictions.
        
        Returns:
            Dictionary of trajectory error metrics
        """
        predictions = []
        targets = []
        
        for sample in self.test_dataset:
            pred = self.model.predict_final_state(
                sample['initial_state'],
                sample['k_final']
            )
            
            predictions.append([pred.lambda_tilde, pred.gamma_tilde, pred.mu_tilde])
            
            final = sample['final_state']
            targets.append([final.lambda_tilde, final.gamma_tilde, final.mu_tilde])
        
        pred_array = np.array(predictions)
        target_array = np.array(targets)
        
        return TrajectoryErrorMetrics.compute_all(pred_array, target_array)
    
    def evaluate_fixed_point_classification(self) -> Dict[str, float]:
        """
        Evaluate fixed point classification.
        
        Returns:
            Dictionary of classification metrics
        """
        predictions = []
        targets = []
        
        for sample in self.test_dataset:
            _, confidence = self.model.predict_fixed_point(sample['trajectory'])
            predictions.append(confidence)
            
            is_fp = 1.0 if sample['trajectory'].check_fixed_point() else 0.0
            targets.append(is_fp)
        
        pred_array = np.array(predictions)
        target_array = np.array(targets)
        
        return FixedPointMetrics.compute_all(pred_array, target_array)
    
    def evaluate_action_predictions(self) -> Dict[str, float]:
        """
        Evaluate action functional predictions.
        
        Returns:
            Dictionary with R² and error metrics
        """
        predictions = []
        targets = []
        
        for sample in self.test_dataset:
            pred_action = self.model.predict_action(sample['trajectory'])
            true_action = sample['trajectory'].compute_action()
            
            predictions.append(pred_action)
            targets.append(true_action)
        
        pred_array = np.array(predictions).reshape(-1, 1)
        target_array = np.array(targets).reshape(-1, 1)
        
        return TrajectoryErrorMetrics.compute_all(pred_array, target_array)
    
    def evaluate_speedup(
        self,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate speedup vs numerical RG.
        
        Args:
            num_samples: Number of samples for benchmarking
            
        Returns:
            Dictionary with timing and speedup metrics
        """
        return self.benchmark.compute_speedup(num_samples=num_samples)
    
    def evaluate_all(
        self,
        include_speedup: bool = True,
        speedup_samples: int = 100
    ) -> Dict[str, Dict]:
        """
        Run complete evaluation suite.
        
        Args:
            include_speedup: Whether to include speedup benchmark
            speedup_samples: Number of samples for speedup test
            
        Returns:
            Complete evaluation results
        """
        print("Running complete model evaluation...")
        
        results = {}
        
        # Coupling predictions
        print("  1. Evaluating coupling predictions...")
        results['coupling_predictions'] = self.evaluate_coupling_predictions()
        
        # Fixed point classification
        print("  2. Evaluating fixed point classification...")
        results['fixed_point_classification'] = self.evaluate_fixed_point_classification()
        
        # Action predictions
        print("  3. Evaluating action predictions...")
        results['action_predictions'] = self.evaluate_action_predictions()
        
        # Speedup benchmark
        if include_speedup:
            print(f"  4. Running speedup benchmark ({speedup_samples} samples)...")
            results['speedup_benchmark'] = self.evaluate_speedup(speedup_samples)
        
        print("✓ Evaluation complete!")
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Dict]) -> None:
        """
        Print formatted evaluation report.
        
        Args:
            results: Results from evaluate_all()
        """
        print("\n" + "="*60)
        print("IRH SURROGATE MODEL EVALUATION REPORT")
        print("="*60)
        
        # Coupling predictions
        print("\n1. COUPLING PREDICTIONS")
        print("-" * 40)
        coupling = results['coupling_predictions']
        print(f"  MSE:        {coupling['mse']:.6f}")
        print(f"  MAE:        {coupling['mae']:.6f}")
        print(f"  MAPE:       {coupling['mape']:.2f}%")
        print(f"  R²:         {coupling['r_squared']:.6f}")
        
        # Fixed point classification
        print("\n2. FIXED POINT CLASSIFICATION")
        print("-" * 40)
        fp = results['fixed_point_classification']
        print(f"  Accuracy:   {fp['accuracy']:.4f}")
        print(f"  Precision:  {fp['precision']:.4f}")
        print(f"  Recall:     {fp['recall']:.4f}")
        print(f"  F1 Score:   {fp['f1_score']:.4f}")
        
        # Action predictions
        print("\n3. ACTION PREDICTIONS")
        print("-" * 40)
        action = results['action_predictions']
        print(f"  MSE:        {action['mse']:.6f}")
        print(f"  MAE:        {action['mae']:.6f}")
        print(f"  R²:         {action['r_squared']:.6f}")
        
        # Speedup
        if 'speedup_benchmark' in results:
            print("\n4. SPEEDUP BENCHMARK")
            print("-" * 40)
            speedup = results['speedup_benchmark']
            print(f"  Samples:              {speedup['num_samples']}")
            print(f"  ML time:              {speedup['ml_time_seconds']:.4f}s")
            print(f"  Numerical RG time:    {speedup['numerical_time_seconds']:.4f}s")
            print(f"  SPEEDUP FACTOR:       {speedup['speedup_factor']:.1f}x")
            print(f"  ML per sample:        {speedup['ml_time_per_sample_ms']:.2f}ms")
            print(f"  Numerical per sample: {speedup['numerical_time_per_sample_ms']:.2f}ms")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    print("Testing Evaluation Metrics...")
    
    # Test 1: Trajectory error metrics
    print("\n1. Testing TrajectoryErrorMetrics...")
    pred = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    target = np.array([[10.5, 20.5, 30.5], [11.5, 21.5, 31.5]])
    
    metrics = TrajectoryErrorMetrics.compute_all(pred, target)
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  R²:   {metrics['r_squared']:.4f}")
    
    # Test 2: Fixed point metrics
    print("\n2. Testing FixedPointMetrics...")
    pred_prob = np.array([0.8, 0.3, 0.6, 0.9, 0.2])
    target_labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    
    fp_metrics = FixedPointMetrics.compute_all(pred_prob, target_labels)
    print(f"  Accuracy:  {fp_metrics['accuracy']:.4f}")
    print(f"  Precision: {fp_metrics['precision']:.4f}")
    print(f"  Recall:    {fp_metrics['recall']:.4f}")
    print(f"  F1 Score:  {fp_metrics['f1_score']:.4f}")
    
    # Test 3: Model evaluator (demonstration)
    print("\n3. Testing ModelEvaluator setup...")
    from models import IRHTransformer
    from training import RGTrajectoryDataset
    from engines import ResonanceEngine
    
    model = IRHTransformer(embed_dim=64)
    test_data = RGTrajectoryDataset(num_samples=10, random_seed=42)
    engine = ResonanceEngine()
    
    evaluator = ModelEvaluator(model, test_data, engine)
    print(f"  ✓ Evaluator created")
    print(f"    Test samples: {len(test_data)}")
    
    # Note: Full evaluation would require trained model
    print("\n✅ All evaluation metrics tested!")
    print("\nNOTE: Full evaluation requires a trained model.")
    print("      Run evaluate_all() on trained model for complete report.")
