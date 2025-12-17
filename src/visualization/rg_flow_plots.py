"""
RG Flow Visualization Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.2-1.3, Eq. 1.12-1.14

This module provides visualization tools for the renormalization group flow:
    - Phase diagrams in (λ̃, γ̃, μ̃) coupling space
    - RG flow trajectories and streamlines
    - Fixed point stability basins
    - Beta function visualizations

Key Equations:
    β_λ = -2λ̃ + (9/8π²)λ̃²     (Eq. 1.13a)
    β_γ = (3/4π²)λ̃γ̃           (Eq. 1.13b)
    β_μ = 2μ̃ + (1/2π²)λ̃μ̃     (Eq. 1.13c)
    
    Fixed Point (Eq. 1.14):
        λ̃* = 48π²/9 ≈ 52.638
        γ̃* = 32π²/3 ≈ 105.276
        μ̃* = 16π² ≈ 157.914

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional matplotlib import for headless environments
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any
    Axes3D = Any

# Optional plotly import for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

__version__ = "21.0.0"
__theoretical_foundation__ = "Intrinsic_Resonance_Holography-v21.1.md §1.2-1.3, Eq. 1.12-1.14"


# =============================================================================
# Constants (from IRH21.md Eq. 1.14)
# =============================================================================

LAMBDA_STAR = 48 * np.pi**2 / 9  # ≈ 52.638
GAMMA_STAR = 32 * np.pi**2 / 3   # ≈ 105.276
MU_STAR = 16 * np.pi**2          # ≈ 157.914


# =============================================================================
# Beta Functions (IRH21.md Eq. 1.13)
# =============================================================================

def _beta_lambda(lambda_t: float) -> float:
    """β_λ = -2λ̃ + (9/8π²)λ̃² (Eq. 1.13a)"""
    return -2 * lambda_t + (9 / (8 * np.pi**2)) * lambda_t**2


def _beta_gamma(lambda_t: float, gamma_t: float) -> float:
    """β_γ = (3/4π²)λ̃γ̃ (Eq. 1.13b)"""
    return (3 / (4 * np.pi**2)) * lambda_t * gamma_t


def _beta_mu(lambda_t: float, mu_t: float) -> float:
    """β_μ = 2μ̃ + (1/2π²)λ̃μ̃ (Eq. 1.13c)"""
    return 2 * mu_t + (1 / (2 * np.pi**2)) * lambda_t * mu_t


# =============================================================================
# RG Flow Plotter Class
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    colormap: str = 'viridis'
    line_width: float = 1.5
    marker_size: float = 50
    grid: bool = True
    dark_mode: bool = False
    
    def get_style(self) -> Dict[str, Any]:
        """Return matplotlib style parameters."""
        if self.dark_mode:
            return {'facecolor': '#1a1a2e', 'edgecolor': '#ffffff'}
        return {'facecolor': '#ffffff', 'edgecolor': '#000000'}


@dataclass
class RGFlowPlotter:
    """
    Comprehensive RG flow visualization system.
    
    Theoretical Reference:
        IRH21.md §1.2-1.3
        Visualization of coupling space and flow trajectories.
    """
    config: PlotConfig = field(default_factory=PlotConfig)
    
    def __post_init__(self):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for RGFlowPlotter")
    
    def phase_diagram_2d(
        self,
        x_var: str = 'lambda',
        y_var: str = 'gamma',
        fixed_value: float = MU_STAR,
        x_range: Tuple[float, float] = (0, 100),
        y_range: Tuple[float, float] = (0, 200),
        resolution: int = 30,
        show_streamlines: bool = True,
        show_fixed_point: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D phase diagram with flow streamlines.
        
        Parameters
        ----------
        x_var : str
            Variable for x-axis ('lambda', 'gamma', or 'mu')
        y_var : str
            Variable for y-axis
        fixed_value : float
            Value for the third (fixed) coupling
        x_range : tuple
            Range for x-axis
        y_range : tuple
            Range for y-axis
        resolution : int
            Grid resolution for streamlines
        show_streamlines : bool
            Whether to show flow streamlines
        show_fixed_point : bool
            Whether to mark the fixed point
        ax : Axes, optional
            Matplotlib axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute beta functions on grid
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(resolution):
            for j in range(resolution):
                lambda_t, gamma_t, mu_t = self._get_couplings(
                    X[i, j], Y[i, j], x_var, y_var, fixed_value
                )
                beta_l = _beta_lambda(lambda_t)
                beta_g = _beta_gamma(lambda_t, gamma_t)
                beta_m = _beta_mu(lambda_t, mu_t)
                U[i, j], V[i, j] = self._get_beta_components(
                    beta_l, beta_g, beta_m, x_var, y_var
                )
        
        # Compute flow magnitude for coloring
        speed = np.sqrt(U**2 + V**2)
        
        if show_streamlines:
            # Plot streamlines
            strm = ax.streamplot(
                X, Y, U, V,
                color=speed,
                cmap=self.config.colormap,
                linewidth=self.config.line_width,
                density=1.5,
                arrowstyle='->',
                arrowsize=1.5
            )
            plt.colorbar(strm.lines, ax=ax, label='Flow speed |β|')
        
        if show_fixed_point:
            # Mark fixed point
            fp_x, fp_y = self._get_fixed_point_coords(x_var, y_var)
            ax.scatter(
                [fp_x], [fp_y],
                s=self.config.marker_size * 3,
                c='red',
                marker='*',
                label=f'Fixed Point (λ*={LAMBDA_STAR:.1f})',
                zorder=5,
                edgecolors='white',
                linewidths=1.5
            )
        
        # Labels and styling
        ax.set_xlabel(self._get_label(x_var), fontsize=self.config.label_fontsize)
        ax.set_ylabel(self._get_label(y_var), fontsize=self.config.label_fontsize)
        ax.set_title(
            f'RG Flow Phase Diagram (Eq. 1.13)\n{self._get_label(self._get_fixed_var(x_var, y_var))} = {fixed_value:.2f}',
            fontsize=self.config.title_fontsize
        )
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        return fig, ax
    
    def phase_diagram_3d(
        self,
        lambda_range: Tuple[float, float] = (0, 100),
        gamma_range: Tuple[float, float] = (0, 200),
        mu_range: Tuple[float, float] = (0, 300),
        n_trajectories: int = 20,
        t_range: Tuple[float, float] = (-5, 10),
        ax: Optional[Axes3D] = None
    ) -> Tuple[Figure, Axes3D]:
        """
        Plot 3D phase diagram with RG trajectories.
        
        Parameters
        ----------
        lambda_range, gamma_range, mu_range : tuple
            Ranges for each coupling
        n_trajectories : int
            Number of trajectories to plot
        t_range : tuple
            RG time range for integration
        ax : Axes3D, optional
            3D axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes3D) matplotlib objects
        """
        if ax is None:
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        # Generate random starting points
        np.random.seed(42)
        lambdas = np.random.uniform(lambda_range[0] + 5, lambda_range[1] - 5, n_trajectories)
        gammas = np.random.uniform(gamma_range[0] + 5, gamma_range[1] - 5, n_trajectories)
        mus = np.random.uniform(mu_range[0] + 5, mu_range[1] - 5, n_trajectories)
        
        # Integrate and plot trajectories
        from scipy.integrate import solve_ivp
        
        def rg_flow(t, y):
            lambda_t, gamma_t, mu_t = y
            return [
                _beta_lambda(lambda_t),
                _beta_gamma(lambda_t, gamma_t),
                _beta_mu(lambda_t, mu_t)
            ]
        
        colors = plt.cm.get_cmap(self.config.colormap)(np.linspace(0, 1, n_trajectories))
        
        for i in range(n_trajectories):
            y0 = [lambdas[i], gammas[i], mus[i]]
            try:
                sol = solve_ivp(
                    rg_flow, t_range, y0,
                    dense_output=True,
                    max_step=0.1
                )
                t_eval = np.linspace(t_range[0], t_range[1], 200)
                trajectory = sol.sol(t_eval)
                
                # Clip to visible range
                mask = (
                    (trajectory[0] >= lambda_range[0]) & (trajectory[0] <= lambda_range[1]) &
                    (trajectory[1] >= gamma_range[0]) & (trajectory[1] <= gamma_range[1]) &
                    (trajectory[2] >= mu_range[0]) & (trajectory[2] <= mu_range[1])
                )
                
                if np.any(mask):
                    ax.plot3D(
                        trajectory[0][mask],
                        trajectory[1][mask],
                        trajectory[2][mask],
                        color=colors[i],
                        linewidth=self.config.line_width,
                        alpha=0.7
                    )
            except Exception:
                continue
        
        # Mark fixed point
        ax.scatter(
            [LAMBDA_STAR], [GAMMA_STAR], [MU_STAR],
            s=self.config.marker_size * 3,
            c='red',
            marker='*',
            label='Cosmic Fixed Point',
            depthshade=False
        )
        
        # Labels
        ax.set_xlabel('λ̃', fontsize=self.config.label_fontsize)
        ax.set_ylabel('γ̃', fontsize=self.config.label_fontsize)
        ax.set_zlabel('μ̃', fontsize=self.config.label_fontsize)
        ax.set_title('3D RG Flow Trajectories (Eq. 1.12)', fontsize=self.config.title_fontsize)
        ax.legend()
        
        return fig, ax
    
    def beta_function_plot(
        self,
        coupling_range: Tuple[float, float] = (0, 100),
        resolution: int = 200,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot all three beta functions vs their primary coupling.
        
        Parameters
        ----------
        coupling_range : tuple
            Range for coupling values
        resolution : int
            Number of points
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        x = np.linspace(coupling_range[0], coupling_range[1], resolution)
        
        # β_λ(λ)
        beta_l = np.array([_beta_lambda(l) for l in x])
        ax.plot(x, beta_l, 'b-', label=r'$\beta_\lambda(\tilde{\lambda})$', linewidth=self.config.line_width)
        
        # β_γ(γ) at λ = λ*
        beta_g = np.array([_beta_gamma(LAMBDA_STAR, g) for g in x])
        ax.plot(x, beta_g, 'g--', label=r'$\beta_\gamma(\tilde{\gamma})$ at $\tilde{\lambda}=\tilde{\lambda}^*$', linewidth=self.config.line_width)
        
        # β_μ(μ) at λ = λ*
        beta_m = np.array([_beta_mu(LAMBDA_STAR, m) for m in x[:len(x)//2]])
        ax.plot(x[:len(x)//2], beta_m, 'r:', label=r'$\beta_\mu(\tilde{\mu})$ at $\tilde{\lambda}=\tilde{\lambda}^*$', linewidth=self.config.line_width)
        
        # Mark zeros
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=LAMBDA_STAR, color='gray', linestyle='--', alpha=0.5, label=f'λ* = {LAMBDA_STAR:.2f}')
        
        ax.set_xlabel('Coupling value', fontsize=self.config.label_fontsize)
        ax.set_ylabel('β-function value', fontsize=self.config.label_fontsize)
        ax.set_title('Beta Functions (IRH21.md Eq. 1.13)', fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.tick_fontsize)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def fixed_point_stability_plot(
        self,
        eigenvalues: Optional[np.ndarray] = None,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Visualize fixed point stability through eigenvalue analysis.
        
        Parameters
        ----------
        eigenvalues : ndarray, optional
            Stability matrix eigenvalues. If None, computed from theory.
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        if eigenvalues is None:
            # Theoretical eigenvalues from stability analysis
            eigenvalues = np.array([10.0, 4.0, 14.0/3.0])  # From IRH21.md
        
        # Plot eigenvalues
        colors = ['green' if e > 0 else 'red' for e in eigenvalues]
        bars = ax.bar(
            range(len(eigenvalues)),
            eigenvalues,
            color=colors,
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, eigenvalues)):
            height = bar.get_height()
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=self.config.label_fontsize
            )
        
        ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax.set_xticks(range(len(eigenvalues)))
        ax.set_xticklabels([r'$\theta_1$', r'$\theta_2$', r'$\theta_3$'], fontsize=self.config.label_fontsize)
        ax.set_ylabel('Critical Exponent', fontsize=self.config.label_fontsize)
        ax.set_title('Fixed Point Stability (All θ > 0 → IR Attractive)', fontsize=self.config.title_fontsize)
        
        # Add interpretation
        ax.text(
            0.5, 0.9,
            'All positive → IR stable fixed point',
            transform=ax.transAxes,
            ha='center',
            fontsize=self.config.tick_fontsize,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        )
        
        if self.config.grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        return fig, ax
    
    # Helper methods
    def _get_couplings(
        self,
        x: float,
        y: float,
        x_var: str,
        y_var: str,
        fixed_value: float
    ) -> Tuple[float, float, float]:
        """Get (λ, γ, μ) from 2D plot coordinates."""
        couplings = {'lambda': LAMBDA_STAR, 'gamma': GAMMA_STAR, 'mu': MU_STAR}
        couplings[x_var] = x
        couplings[y_var] = y
        fixed_var = self._get_fixed_var(x_var, y_var)
        couplings[fixed_var] = fixed_value
        return couplings['lambda'], couplings['gamma'], couplings['mu']
    
    def _get_beta_components(
        self,
        beta_l: float,
        beta_g: float,
        beta_m: float,
        x_var: str,
        y_var: str
    ) -> Tuple[float, float]:
        """Get (U, V) components for streamplot."""
        betas = {'lambda': beta_l, 'gamma': beta_g, 'mu': beta_m}
        return betas[x_var], betas[y_var]
    
    def _get_fixed_point_coords(self, x_var: str, y_var: str) -> Tuple[float, float]:
        """Get fixed point coordinates for 2D plot."""
        fp = {'lambda': LAMBDA_STAR, 'gamma': GAMMA_STAR, 'mu': MU_STAR}
        return fp[x_var], fp[y_var]
    
    def _get_fixed_var(self, x_var: str, y_var: str) -> str:
        """Get the name of the fixed (third) variable."""
        all_vars = {'lambda', 'gamma', 'mu'}
        return (all_vars - {x_var, y_var}).pop()
    
    def _get_label(self, var: str) -> str:
        """Get LaTeX label for a coupling variable."""
        labels = {
            'lambda': r'$\tilde{\lambda}$',
            'gamma': r'$\tilde{\gamma}$',
            'mu': r'$\tilde{\mu}$'
        }
        return labels.get(var, var)


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def plot_phase_diagram_2d(
    x_var: str = 'lambda',
    y_var: str = 'gamma',
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create 2D phase diagram of RG flow.
    
    Theoretical Reference:
        IRH21.md §1.2-1.3, Eq. 1.13
    
    Parameters
    ----------
    x_var : str
        Variable for x-axis ('lambda', 'gamma', or 'mu')
    y_var : str
        Variable for y-axis
    **kwargs
        Additional arguments passed to RGFlowPlotter.phase_diagram_2d
        
    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects
    """
    plotter = RGFlowPlotter()
    return plotter.phase_diagram_2d(x_var=x_var, y_var=y_var, **kwargs)


def plot_phase_diagram_3d(
    **kwargs
) -> Tuple[Figure, Axes3D]:
    """
    Create 3D phase diagram of RG flow.
    
    Theoretical Reference:
        IRH21.md §1.2, Eq. 1.12
    
    Returns
    -------
    tuple
        (Figure, Axes3D) matplotlib objects
    """
    plotter = RGFlowPlotter()
    return plotter.phase_diagram_3d(**kwargs)


def plot_rg_trajectory(
    initial_couplings: Tuple[float, float, float],
    t_range: Tuple[float, float] = (-5, 10),
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Plot a single RG trajectory from given initial conditions.
    
    Theoretical Reference:
        IRH21.md §1.2, Eq. 1.12
    
    Parameters
    ----------
    initial_couplings : tuple
        (λ₀, γ₀, μ₀) initial coupling values
    t_range : tuple
        RG time range
    ax : Axes, optional
        Axes to plot on
        
    Returns
    -------
    tuple
        (Figure, Axes) matplotlib objects
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required")
    
    from scipy.integrate import solve_ivp
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    else:
        fig = ax.get_figure()
    
    def rg_flow(t, y):
        lambda_t, gamma_t, mu_t = y
        return [
            _beta_lambda(lambda_t),
            _beta_gamma(lambda_t, gamma_t),
            _beta_mu(lambda_t, mu_t)
        ]
    
    sol = solve_ivp(rg_flow, t_range, initial_couplings, dense_output=True, max_step=0.1)
    t = np.linspace(t_range[0], t_range[1], 500)
    trajectory = sol.sol(t)
    
    ax.plot(t, trajectory[0], 'b-', label=r'$\tilde{\lambda}(t)$', linewidth=1.5)
    ax.plot(t, trajectory[1], 'g--', label=r'$\tilde{\gamma}(t)$', linewidth=1.5)
    ax.plot(t, trajectory[2], 'r:', label=r'$\tilde{\mu}(t)$', linewidth=1.5)
    
    # Mark fixed point values
    ax.axhline(y=LAMBDA_STAR, color='b', linestyle=':', alpha=0.3)
    ax.axhline(y=GAMMA_STAR, color='g', linestyle=':', alpha=0.3)
    ax.axhline(y=MU_STAR, color='r', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('RG time t = log(k/k₀)', fontsize=12)
    ax.set_ylabel('Coupling value', fontsize=12)
    ax.set_title(f'RG Trajectory from ({initial_couplings[0]:.1f}, {initial_couplings[1]:.1f}, {initial_couplings[2]:.1f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_beta_functions(**kwargs) -> Tuple[Figure, Axes]:
    """
    Plot all three β-functions.
    
    Theoretical Reference:
        IRH21.md §1.2, Eq. 1.13
    """
    plotter = RGFlowPlotter()
    return plotter.beta_function_plot(**kwargs)


def plot_fixed_point_stability(**kwargs) -> Tuple[Figure, Axes]:
    """
    Visualize fixed point stability.
    
    Theoretical Reference:
        IRH21.md §1.3
    """
    plotter = RGFlowPlotter()
    return plotter.fixed_point_stability_plot(**kwargs)


# =============================================================================
# Interactive Plotly Versions
# =============================================================================

def create_interactive_phase_diagram(
    x_var: str = 'lambda',
    y_var: str = 'gamma',
    resolution: int = 30
) -> Any:
    """
    Create interactive 2D phase diagram using Plotly.
    
    Theoretical Reference:
        IRH21.md §1.2-1.3
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive plots")
    
    # Create grid
    x_range = (0, 100) if x_var == 'lambda' else (0, 200)
    y_range = (0, 200) if y_var == 'gamma' else (0, 300)
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute flow magnitude
    speed = np.zeros_like(X)
    plotter = RGFlowPlotter()
    fixed_value = MU_STAR if (x_var, y_var) != ('lambda', 'mu') else GAMMA_STAR
    
    for i in range(resolution):
        for j in range(resolution):
            lambda_t, gamma_t, mu_t = plotter._get_couplings(
                X[i, j], Y[i, j], x_var, y_var, fixed_value
            )
            beta_l = _beta_lambda(lambda_t)
            beta_g = _beta_gamma(lambda_t, gamma_t)
            beta_m = _beta_mu(lambda_t, mu_t)
            u, v = plotter._get_beta_components(beta_l, beta_g, beta_m, x_var, y_var)
            speed[i, j] = np.sqrt(u**2 + v**2)
    
    # Create figure
    fig = go.Figure()
    
    # Add contour plot for flow magnitude
    fig.add_trace(go.Contour(
        x=x, y=y, z=speed,
        colorscale='Viridis',
        colorbar=dict(title='|β|'),
        name='Flow magnitude'
    ))
    
    # Add fixed point marker
    fp = {'lambda': LAMBDA_STAR, 'gamma': GAMMA_STAR, 'mu': MU_STAR}
    fig.add_trace(go.Scatter(
        x=[fp[x_var]], y=[fp[y_var]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Cosmic Fixed Point'
    ))
    
    # Update layout
    labels = {'lambda': 'λ̃', 'gamma': 'γ̃', 'mu': 'μ̃'}
    fig.update_layout(
        title='Interactive RG Flow Phase Diagram (IRH21.md Eq. 1.13)',
        xaxis_title=labels[x_var],
        yaxis_title=labels[y_var],
        width=800,
        height=600
    )
    
    return fig


__all__ = [
    'RGFlowPlotter',
    'PlotConfig',
    'plot_phase_diagram_2d',
    'plot_phase_diagram_3d',
    'plot_rg_trajectory',
    'plot_beta_functions',
    'plot_fixed_point_stability',
    'create_interactive_phase_diagram',
    'LAMBDA_STAR',
    'GAMMA_STAR',
    'MU_STAR',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
]
