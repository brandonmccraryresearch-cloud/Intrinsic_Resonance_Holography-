"""
IRH Desktop - Visualization Plots

Matplotlib-based plotting for IRH computations with
interactive features when used with PyQt6.

This module implements Phase 5 of the DEB_PACKAGE_ROADMAP.md:
- Matplotlib integration
- Interactive plot widgets
- RG flow visualization
- Spectral dimension plots

Theoretical Foundation:
    IRH21.md - Visualizes equations from §1-2

Author: Brandon D. McCrary
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = object
    Axes = object


# Theoretical constants (from IRH21.md)
FIXED_POINT_LAMBDA = 48 * np.pi**2 / 9   # ≈ 52.638
FIXED_POINT_GAMMA = 32 * np.pi**2 / 3    # ≈ 105.276  
FIXED_POINT_MU = 16 * np.pi**2           # ≈ 157.914
C_H = 3 * FIXED_POINT_LAMBDA / (2 * FIXED_POINT_GAMMA)  # ≈ 0.04594


@dataclass
class PlotStyle:
    """
    Style configuration for IRH plots.
    
    Attributes
    ----------
    dark_mode : bool
        Use dark theme
    dpi : int
        Figure DPI
    figsize : Tuple[float, float]
        Figure size in inches
    line_width : float
        Default line width
    marker_size : float
        Default marker size
    font_size : int
        Base font size
    title_size : int
        Title font size
    colors : List[str]
        Color palette
    """
    dark_mode: bool = True
    dpi: int = 100
    figsize: Tuple[float, float] = (10, 6)
    line_width: float = 2.0
    marker_size: float = 8.0
    font_size: int = 11
    title_size: int = 14
    colors: List[str] = field(default_factory=lambda: [
        '#2196f3', '#4caf50', '#ff9800', '#e91e63',
        '#9c27b0', '#00bcd4', '#8bc34a', '#ff5722'
    ])
    
    def apply(self, fig: 'Figure' = None) -> None:
        """Apply style settings to matplotlib."""
        if not HAS_MATPLOTLIB:
            return
        
        if self.dark_mode:
            plt.style.use('dark_background')
            # Additional dark mode settings
            plt.rcParams['figure.facecolor'] = '#1e1e1e'
            plt.rcParams['axes.facecolor'] = '#252526'
            plt.rcParams['axes.edgecolor'] = '#3a3a3a'
            plt.rcParams['grid.color'] = '#3a3a3a'
            plt.rcParams['text.color'] = '#d4d4d4'
            plt.rcParams['axes.labelcolor'] = '#d4d4d4'
            plt.rcParams['xtick.color'] = '#d4d4d4'
            plt.rcParams['ytick.color'] = '#d4d4d4'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['lines.linewidth'] = self.line_width
        plt.rcParams['lines.markersize'] = self.marker_size
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['axes.titlesize'] = self.title_size


@dataclass
class RGTrajectory:
    """
    A renormalization group trajectory.
    
    Attributes
    ----------
    t : np.ndarray
        RG time parameter
    lambda_tilde : np.ndarray
        Dimensionless λ coupling
    gamma_tilde : np.ndarray
        Dimensionless γ coupling
    mu_tilde : np.ndarray
        Dimensionless μ coupling
    label : str
        Trajectory label
    """
    t: np.ndarray
    lambda_tilde: np.ndarray
    gamma_tilde: np.ndarray
    mu_tilde: np.ndarray
    label: str = "RG Trajectory"


class RGFlowPlot:
    """
    Interactive RG flow visualization.
    
    Creates plots showing RG trajectories converging to the
    Cosmic Fixed Point (Eq. 1.14).
    
    Parameters
    ----------
    style : PlotStyle
        Plot style configuration
        
    Examples
    --------
    >>> plot = RGFlowPlot()
    >>> trajectory = RGTrajectory(t, lambda_vals, gamma_vals, mu_vals)
    >>> plot.add_trajectory(trajectory)
    >>> plot.mark_fixed_point()
    >>> fig = plot.get_figure()
    
    Theoretical Foundation
    ----------------------
    Visualizes RG flow from IRH21.md §1.2-1.3
    - Beta functions: Eq. 1.13
    - Cosmic Fixed Point: Eq. 1.14
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize RG Flow Plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )
        
        self.style = style or PlotStyle()
        self.style.apply()
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle(
            'RG Flow to Cosmic Fixed Point (IRH21.md §1.2-1.3)',
            fontsize=self.style.title_size
        )
        
        # Trajectory storage
        self.trajectories: List[RGTrajectory] = []
        
        # Setup axes
        self._setup_axes()
    
    def _setup_axes(self) -> None:
        """Configure the plot axes."""
        # λ̃ vs t
        ax = self.axes[0, 0]
        ax.set_xlabel('RG Time t')
        ax.set_ylabel('λ̃')
        ax.set_title('λ̃(t) → λ̃* = 48π²/9')
        ax.grid(True, alpha=0.3)
        
        # γ̃ vs t
        ax = self.axes[0, 1]
        ax.set_xlabel('RG Time t')
        ax.set_ylabel('γ̃')
        ax.set_title('γ̃(t) → γ̃* = 32π²/3')
        ax.grid(True, alpha=0.3)
        
        # μ̃ vs t
        ax = self.axes[1, 0]
        ax.set_xlabel('RG Time t')
        ax.set_ylabel('μ̃')
        ax.set_title('μ̃(t) → μ̃* = 16π²')
        ax.grid(True, alpha=0.3)
        
        # Phase portrait (λ̃ vs γ̃)
        ax = self.axes[1, 1]
        ax.set_xlabel('λ̃')
        ax.set_ylabel('γ̃')
        ax.set_title('Phase Portrait (λ̃, γ̃)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def add_trajectory(
        self,
        trajectory: RGTrajectory,
        color: Optional[str] = None
    ) -> None:
        """
        Add a trajectory to the plot.
        
        Parameters
        ----------
        trajectory : RGTrajectory
            Trajectory data
        color : str, optional
            Line color
        """
        self.trajectories.append(trajectory)
        
        idx = len(self.trajectories) - 1
        c = color or self.style.colors[idx % len(self.style.colors)]
        
        # Plot on each axis
        self.axes[0, 0].plot(trajectory.t, trajectory.lambda_tilde, 
                            color=c, label=trajectory.label)
        self.axes[0, 1].plot(trajectory.t, trajectory.gamma_tilde, 
                            color=c, label=trajectory.label)
        self.axes[1, 0].plot(trajectory.t, trajectory.mu_tilde, 
                            color=c, label=trajectory.label)
        self.axes[1, 1].plot(trajectory.lambda_tilde, trajectory.gamma_tilde, 
                            color=c, label=trajectory.label)
    
    def mark_fixed_point(
        self,
        show_values: bool = True,
        marker: str = '*',
        size: float = 200
    ) -> None:
        """
        Mark the Cosmic Fixed Point location.
        
        Parameters
        ----------
        show_values : bool
            Show numerical values
        marker : str
            Marker style (matplotlib marker code)
        size : float
            Marker size
        """
        fp_color = '#ff5722'  # Orange-red for fixed point
        
        # Mark on time plots as horizontal lines
        self.axes[0, 0].axhline(y=FIXED_POINT_LAMBDA, color=fp_color, 
                                linestyle='--', alpha=0.7, label='λ̃*')
        self.axes[0, 1].axhline(y=FIXED_POINT_GAMMA, color=fp_color, 
                                linestyle='--', alpha=0.7, label='γ̃*')
        self.axes[1, 0].axhline(y=FIXED_POINT_MU, color=fp_color, 
                                linestyle='--', alpha=0.7, label='μ̃*')
        
        # Mark on phase portrait
        self.axes[1, 1].scatter([FIXED_POINT_LAMBDA], [FIXED_POINT_GAMMA],
                               marker=marker, s=size, color=fp_color,
                               zorder=10, label='Fixed Point')
        
        if show_values:
            # Add text annotations
            self.axes[0, 0].annotate(f'λ̃* = {FIXED_POINT_LAMBDA:.2f}',
                                    xy=(0.02, 0.95), xycoords='axes fraction',
                                    fontsize=9, color=fp_color)
            self.axes[0, 1].annotate(f'γ̃* = {FIXED_POINT_GAMMA:.2f}',
                                    xy=(0.02, 0.95), xycoords='axes fraction',
                                    fontsize=9, color=fp_color)
            self.axes[1, 0].annotate(f'μ̃* = {FIXED_POINT_MU:.2f}',
                                    xy=(0.02, 0.95), xycoords='axes fraction',
                                    fontsize=9, color=fp_color)
            self.axes[1, 1].annotate(f'(λ̃*, γ̃*)',
                                    xy=(FIXED_POINT_LAMBDA, FIXED_POINT_GAMMA),
                                    xytext=(10, 10), textcoords='offset points',
                                    fontsize=9, color=fp_color)
    
    def add_reference(self, reference: str = "IRH21.md §1.2-1.3, Eq. 1.14") -> None:
        """Add theoretical reference to plot."""
        self.fig.text(0.02, 0.02, f'Reference: {reference}',
                     fontsize=8, alpha=0.7)
    
    def add_legend(self) -> None:
        """Add legends to all subplots."""
        for ax in self.axes.flat:
            ax.legend(loc='best', fontsize=8)
    
    def get_figure(self) -> 'Figure':
        """Get the matplotlib figure."""
        return self.fig
    
    def save(self, path: str, **kwargs) -> None:
        """
        Save plot to file.
        
        Parameters
        ----------
        path : str
            Output file path
        **kwargs
            Additional arguments for savefig
        """
        self.fig.savefig(path, dpi=self.style.dpi, bbox_inches='tight', **kwargs)
        logger.info(f"Saved RG flow plot to {path}")
    
    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)


class SpectralDimensionPlot:
    """
    Spectral dimension flow visualization.
    
    Shows d_spec(k) flowing from 2 in UV to 4 in IR,
    verifying Theorem 2.1.
    
    Parameters
    ----------
    style : PlotStyle
        Plot style configuration
        
    Examples
    --------
    >>> plot = SpectralDimensionPlot()
    >>> plot.plot_flow(k_values, d_spec_values)
    >>> plot.mark_limits()
    >>> fig = plot.get_figure()
    
    Theoretical Foundation
    ----------------------
    Visualizes spectral dimension from IRH21.md §2.1.2
    - d_spec flow: Eq. 2.8-2.9
    - Theorem 2.1: d_spec → 4 exactly
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize Spectral Dimension Plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization.")
        
        self.style = style or PlotStyle()
        self.style.apply()
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle(
            'Spectral Dimension Flow d_spec(k) (IRH21.md §2.1.2, Theorem 2.1)',
            fontsize=self.style.title_size
        )
        
        self._setup_axes()
    
    def _setup_axes(self) -> None:
        """Configure plot axes."""
        self.ax.set_xlabel('log₁₀(k/k_IR)')
        self.ax.set_ylabel('d_spec')
        self.ax.set_ylim(1.5, 4.5)
        self.ax.grid(True, alpha=0.3)
    
    def plot_flow(
        self,
        k_values: np.ndarray,
        d_spec_values: np.ndarray,
        label: str = "d_spec(k)",
        color: Optional[str] = None
    ) -> None:
        """
        Plot spectral dimension flow.
        
        Parameters
        ----------
        k_values : np.ndarray
            Momentum scale values
        d_spec_values : np.ndarray
            Spectral dimension values
        label : str
            Plot label
        color : str, optional
            Line color
        """
        c = color or self.style.colors[0]
        self.ax.plot(k_values, d_spec_values, color=c, 
                    linewidth=self.style.line_width, label=label)
    
    def mark_limits(self) -> None:
        """Mark UV and IR limits."""
        # UV limit: d_spec = 2
        self.ax.axhline(y=2.0, color='#ff9800', linestyle='--', 
                       alpha=0.7, label='UV limit (d=2)')
        
        # IR limit: d_spec = 4
        self.ax.axhline(y=4.0, color='#4caf50', linestyle='--', 
                       alpha=0.7, label='IR limit (d=4)')
        
        # Add annotations
        self.ax.annotate('d_spec → 2 (UV)',
                        xy=(0.95, 2.0), xycoords=('axes fraction', 'data'),
                        ha='right', fontsize=9, color='#ff9800')
        self.ax.annotate('d_spec → 4 (IR, Theorem 2.1)',
                        xy=(0.95, 4.0), xycoords=('axes fraction', 'data'),
                        ha='right', fontsize=9, color='#4caf50')
    
    def add_crossover_region(
        self,
        k_start: float,
        k_end: float,
        label: str = "Crossover"
    ) -> None:
        """Highlight the dimensional crossover region."""
        self.ax.axvspan(k_start, k_end, alpha=0.2, color='#9c27b0',
                       label=label)
    
    def add_reference(self, reference: str = "IRH21.md §2.1.2, Eq. 2.8-2.9") -> None:
        """Add theoretical reference."""
        self.fig.text(0.02, 0.02, f'Reference: {reference}',
                     fontsize=8, alpha=0.7)
    
    def add_legend(self) -> None:
        """Add legend."""
        self.ax.legend(loc='best')
    
    def get_figure(self) -> 'Figure':
        """Get the matplotlib figure."""
        return self.fig
    
    def save(self, path: str, **kwargs) -> None:
        """Save plot to file."""
        self.fig.savefig(path, dpi=self.style.dpi, bbox_inches='tight', **kwargs)
        logger.info(f"Saved spectral dimension plot to {path}")
    
    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)


class FixedPointPlot:
    """
    Cosmic Fixed Point basin of attraction visualization.
    
    Shows how different initial conditions flow to the
    unique infrared attractor.
    
    Parameters
    ----------
    style : PlotStyle
        Plot style configuration
        
    Theoretical Foundation
    ----------------------
    Visualizes fixed point from IRH21.md §1.3
    - Fixed point: Eq. 1.14
    - Stability analysis: §1.3.2
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """Initialize Fixed Point Plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization.")
        
        self.style = style or PlotStyle()
        self.style.apply()
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.suptitle(
            'Basin of Attraction: Cosmic Fixed Point (IRH21.md §1.3)',
            fontsize=self.style.title_size
        )
        
        self._setup_axes()
    
    def _setup_axes(self) -> None:
        """Configure plot axes."""
        self.ax.set_xlabel('λ̃')
        self.ax.set_ylabel('γ̃')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
    
    def plot_vector_field(
        self,
        lambda_range: Tuple[float, float] = (0, 100),
        gamma_range: Tuple[float, float] = (0, 200),
        n_points: int = 15
    ) -> None:
        """
        Plot beta function vector field.
        
        Parameters
        ----------
        lambda_range : Tuple[float, float]
            Range for λ̃
        gamma_range : Tuple[float, float]
            Range for γ̃
        n_points : int
            Number of grid points per dimension
        """
        # Create grid
        l = np.linspace(*lambda_range, n_points)
        g = np.linspace(*gamma_range, n_points)
        L, G = np.meshgrid(l, g)
        
        # Beta functions (simplified, from Eq. 1.13)
        beta_lambda = -2 * L + (9 / (8 * np.pi**2)) * L**2
        beta_gamma = (3 / (4 * np.pi**2)) * L * G
        
        # Normalize for visualization
        magnitude = np.sqrt(beta_lambda**2 + beta_gamma**2)
        magnitude = np.maximum(magnitude, 1e-10)  # Avoid division by zero
        
        # Plot quiver
        self.ax.quiver(L, G, beta_lambda/magnitude, beta_gamma/magnitude,
                      magnitude, cmap='viridis', alpha=0.7)
    
    def mark_fixed_point(
        self,
        marker: str = '*',
        size: float = 300
    ) -> None:
        """Mark the fixed point location."""
        self.ax.scatter([FIXED_POINT_LAMBDA], [FIXED_POINT_GAMMA],
                       marker=marker, s=size, color='#ff5722',
                       zorder=10, label=f'Fixed Point\n(λ̃*={FIXED_POINT_LAMBDA:.1f}, γ̃*={FIXED_POINT_GAMMA:.1f})')
    
    def add_trajectories(
        self,
        n_trajectories: int = 10,
        t_max: float = 20.0
    ) -> None:
        """Add sample RG trajectories flowing to fixed point."""
        # Generate random initial conditions
        np.random.seed(42)  # Reproducible
        
        for i in range(n_trajectories):
            # Random starting point
            l0 = np.random.uniform(10, 80)
            g0 = np.random.uniform(20, 180)
            
            # Simple RG integration (Euler method for demo)
            dt = 0.01
            t = np.arange(0, t_max, dt)
            
            l_traj = [l0]
            g_traj = [g0]
            
            for _ in t[:-1]:
                l = l_traj[-1]
                g = g_traj[-1]
                
                # Clamp values to prevent overflow
                l = np.clip(l, 0.01, 200)
                g = np.clip(g, 0.01, 400)
                
                # Beta functions
                beta_l = -2 * l + (9 / (8 * np.pi**2)) * l**2
                beta_g = (3 / (4 * np.pi**2)) * l * g
                
                # Update with clamping
                new_l = l + dt * beta_l
                new_g = g + dt * beta_g
                
                # Stop if diverging
                if new_l > 200 or new_g > 400 or new_l < 0 or new_g < 0:
                    break
                
                l_traj.append(new_l)
                g_traj.append(new_g)
            
            c = self.style.colors[i % len(self.style.colors)]
            self.ax.plot(l_traj, g_traj, color=c, alpha=0.6, linewidth=1)
            # Mark starting point
            self.ax.scatter([l0], [g0], marker='o', s=30, color=c, alpha=0.8)
    
    def add_reference(self, reference: str = "IRH21.md §1.3, Eq. 1.14") -> None:
        """Add theoretical reference."""
        self.fig.text(0.02, 0.02, f'Reference: {reference}',
                     fontsize=8, alpha=0.7)
    
    def add_legend(self) -> None:
        """Add legend."""
        self.ax.legend(loc='upper right')
    
    def get_figure(self) -> 'Figure':
        """Get the matplotlib figure."""
        return self.fig
    
    def save(self, path: str, **kwargs) -> None:
        """Save plot to file."""
        self.fig.savefig(path, dpi=self.style.dpi, bbox_inches='tight', **kwargs)
        logger.info(f"Saved fixed point plot to {path}")
    
    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)


# Convenience functions for quick plotting

def create_rg_flow_figure(
    trajectories: Optional[List[RGTrajectory]] = None,
    mark_fp: bool = True,
    style: Optional[PlotStyle] = None
) -> 'Figure':
    """
    Create an RG flow figure.
    
    Parameters
    ----------
    trajectories : List[RGTrajectory], optional
        Trajectories to plot
    mark_fp : bool
        Mark fixed point
    style : PlotStyle, optional
        Style configuration
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    
    plot = RGFlowPlot(style=style)
    
    if trajectories:
        for traj in trajectories:
            plot.add_trajectory(traj)
    
    if mark_fp:
        plot.mark_fixed_point()
    
    plot.add_reference()
    plot.add_legend()
    
    return plot.get_figure()


def create_spectral_dimension_figure(
    k_values: Optional[np.ndarray] = None,
    d_spec_values: Optional[np.ndarray] = None,
    style: Optional[PlotStyle] = None
) -> 'Figure':
    """
    Create a spectral dimension figure.
    
    Parameters
    ----------
    k_values : np.ndarray, optional
        Momentum scale values (defaults to demo data)
    d_spec_values : np.ndarray, optional
        Spectral dimension values
    style : PlotStyle, optional
        Style configuration
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    
    plot = SpectralDimensionPlot(style=style)
    
    # Use demo data if not provided
    if k_values is None:
        k_values = np.linspace(-5, 5, 100)
        # Smooth interpolation from 2 to 4
        d_spec_values = 2 + 2 / (1 + np.exp(k_values))
    
    plot.plot_flow(k_values, d_spec_values)
    plot.mark_limits()
    plot.add_reference()
    plot.add_legend()
    
    return plot.get_figure()


def create_fixed_point_figure(
    show_trajectories: bool = True,
    show_vector_field: bool = True,
    style: Optional[PlotStyle] = None
) -> 'Figure':
    """
    Create a fixed point basin of attraction figure.
    
    Parameters
    ----------
    show_trajectories : bool
        Show sample RG trajectories
    show_vector_field : bool
        Show beta function vector field
    style : PlotStyle, optional
        Style configuration
        
    Returns
    -------
    Figure
        Matplotlib figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required")
    
    plot = FixedPointPlot(style=style)
    
    if show_vector_field:
        plot.plot_vector_field()
    
    if show_trajectories:
        plot.add_trajectories()
    
    plot.mark_fixed_point()
    plot.add_reference()
    plot.add_legend()
    
    return plot.get_figure()


# Demo function
def create_demo_figures(output_dir: str = ".") -> Dict[str, str]:
    """
    Create demo figures and save to disk.
    
    Parameters
    ----------
    output_dir : str
        Directory for output files
        
    Returns
    -------
    Dict[str, str]
        Map of figure names to file paths
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return {}
    
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    # Create demo RG trajectory
    t = np.linspace(-5, 20, 200)
    # Simplified RG evolution toward fixed point
    l = FIXED_POINT_LAMBDA * (1 - 0.8 * np.exp(-0.3 * (t + 5)))
    g = FIXED_POINT_GAMMA * (1 - 0.8 * np.exp(-0.2 * (t + 5)))
    m = FIXED_POINT_MU * (1 - 0.8 * np.exp(-0.25 * (t + 5)))
    
    trajectory = RGTrajectory(t=t, lambda_tilde=l, gamma_tilde=g, 
                              mu_tilde=m, label="Demo Trajectory")
    
    # RG Flow plot
    fig = create_rg_flow_figure([trajectory])
    path = str(output_path / "rg_flow.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    files["rg_flow"] = path
    
    # Spectral dimension plot
    fig = create_spectral_dimension_figure()
    path = str(output_path / "spectral_dimension.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    files["spectral_dimension"] = path
    
    # Fixed point plot
    fig = create_fixed_point_figure()
    path = str(output_path / "fixed_point.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    files["fixed_point"] = path
    
    logger.info(f"Created demo figures in {output_dir}")
    return files
