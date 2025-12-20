"""
Utility modules for IRH ML surrogate.

Provides visualization, configuration management, and format conversion tools.

Key Components:
- visualization: Plotting and visualization utilities
- config: Configuration management and experiment tracking
- graph_conversion: Format conversion (NetworkX, pandas, JSON, CSV)
"""

from .visualization import (
    TrainingVisualizer,
    TrajectoryVisualizer,
    EvaluationVisualizer,
    quick_plot_training,
    quick_plot_trajectory,
    quick_plot_evaluation
)

from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
    IRHConfig,
    ExperimentTracker
)

from .graph_conversion import (
    holographic_state_to_networkx,
    networkx_to_holographic_state,
    trajectory_to_dataframe,
    dataframe_to_trajectory,
    export_trajectory_csv,
    import_trajectory_csv,
    export_trajectory_json,
    import_trajectory_json,
    to_dict,
    from_dict
)

__all__ = [
    # Visualization
    'TrainingVisualizer',
    'TrajectoryVisualizer',
    'EvaluationVisualizer',
    'quick_plot_training',
    'quick_plot_trajectory',
    'quick_plot_evaluation',
    # Configuration
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'LossConfig',
    'EvaluationConfig',
    'IRHConfig',
    'ExperimentTracker',
    # Graph conversion
    'holographic_state_to_networkx',
    'networkx_to_holographic_state',
    'trajectory_to_dataframe',
    'dataframe_to_trajectory',
    'export_trajectory_csv',
    'import_trajectory_csv',
    'export_trajectory_json',
    'import_trajectory_json',
    'to_dict',
    'from_dict',
]
