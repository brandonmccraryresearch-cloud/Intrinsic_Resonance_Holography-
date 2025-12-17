"""
Spectral Dimension Visualization Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §2.1, Eq. 2.8-2.9, Theorem 2.1

This module provides visualization tools for spectral dimension flow:
    - d_spec(k) scale-dependent spectral dimension
    - Flow from UV (~2) to IR (exactly 4)
    - Graviton correction visualization
    - Animated spectral dimension evolution

Key Results:
    d_spec → 4.0000000000(1) exactly (Theorem 2.1)
    d_spec(UV) ~ 2 (dimensional reduction at Planck scale)
    Critical scale k_c where d_spec crosses 3

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any
    FuncAnimation = Any

# Optional plotly import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §2.1, Eq. 2.8-2.9, Theorem 2.1"


# =============================================================================
# Physical Constants
# =============================================================================

# Fixed point values (IRH21.md Eq. 1.14)
LAMBDA_STAR = 48 * np.pi**2 / 9
GAMMA_STAR = 32 * np.pi**2 / 3

# Spectral dimension limits
D_SPEC_UV = 2.0  # UV limit (dimensional reduction)
D_SPEC_IR = 4.0  # IR limit (macroscopic spacetime)
D_SPEC_ONE_LOOP = 42.0 / 11.0  # One-loop approximation without graviton


# =============================================================================
# Spectral Dimension Functions
# =============================================================================

def spectral_dimension(
    k: Union[float, np.ndarray],
    k_c: float = 1.0,
    include_graviton: bool = True
) -> Union[float, np.ndarray]:
    """
    Compute spectral dimension d_spec(k).
    
    Theoretical Reference:
        IRH21.md §2.1, Eq. 2.8-2.9
        
    Parameters
    ----------
    k : float or ndarray
        RG scale (k → 0 is IR, k → ∞ is UV)
    k_c : float
        Critical scale
    include_graviton : bool
        Whether to include graviton corrections (required for d_spec → 4)
        
    Returns
    -------
    float or ndarray
        Spectral dimension at scale k
    """
    k = np.asarray(k)
    
    # Interpolation function between UV and IR
    # d_spec(k) = d_UV + (d_IR - d_UV) * f(k/k_c)
    x = k / k_c
    
    if include_graviton:
        # With graviton correction: asymptotes to exactly 4
        d_ir = D_SPEC_IR
    else:
        # One-loop only: asymptotes to 42/11 ≈ 3.818
        d_ir = D_SPEC_ONE_LOOP
    
    # Smooth interpolation using sigmoid-like function
    # This captures the crossover behavior
    f = 1.0 / (1.0 + x**2)
    
    d_spec = D_SPEC_UV + (d_ir - D_SPEC_UV) * f
    
    return d_spec


def graviton_correction(k: Union[float, np.ndarray], k_c: float = 1.0) -> Union[float, np.ndarray]:
    """
    Compute graviton correction Δ_grav(k) to spectral dimension.
    
    Theoretical Reference:
        IRH21.md §2.1.2
        Graviton propagator correction that ensures d_spec → 4 exactly.
    """
    k = np.asarray(k)
    x = k / k_c
    
    # Graviton correction profile
    # Dominant in IR, vanishes in UV
    delta = (D_SPEC_IR - D_SPEC_ONE_LOOP) / (1.0 + x**2)
    
    return delta


def find_critical_scale(d_target: float = 3.0) -> float:
    """
    Find scale k where d_spec(k) = d_target.
    
    Parameters
    ----------
    d_target : float
        Target spectral dimension (default 3.0 for dimensional crossover)
        
    Returns
    -------
    float
        Critical scale k_c where d_spec = d_target
    """
    from scipy.optimize import brentq
    
    def f(k):
        return spectral_dimension(k) - d_target
    
    # Search in reasonable range
    k_crit = brentq(f, 0.01, 100.0)
    return k_crit


# =============================================================================
# Spectral Dimension Animator Class
# =============================================================================

@dataclass
class SpectralDimensionAnimator:
    """
    Visualization system for spectral dimension flow.
    
    Theoretical Reference:
        IRH21.md §2.1, Theorem 2.1
        d_spec flows from ~2 (UV) to exactly 4 (IR).
    """
    figsize: Tuple[float, float] = (12, 6)
    dpi: int = 100
    colormap: str = 'plasma'
    
    def __post_init__(self):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for SpectralDimensionAnimator")
    
    def plot_flow(
        self,
        k_range: Tuple[float, float] = (0.01, 100),
        n_points: int = 200,
        show_one_loop: bool = True,
        show_graviton: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot spectral dimension as function of RG scale.
        
        Parameters
        ----------
        k_range : tuple
            Range of RG scales
        n_points : int
            Number of points
        show_one_loop : bool
            Show one-loop result (without graviton)
        show_graviton : bool
            Show graviton-corrected result
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_points)
        
        if show_graviton:
            d_grav = spectral_dimension(k, include_graviton=True)
            ax.plot(k, d_grav, 'b-', linewidth=2.5, label=r'$d_{spec}(k)$ with graviton')
        
        if show_one_loop:
            d_one_loop = spectral_dimension(k, include_graviton=False)
            ax.plot(k, d_one_loop, 'r--', linewidth=2, label=r'$d_{spec}(k)$ one-loop only')
        
        # Mark important values
        ax.axhline(y=4.0, color='green', linestyle=':', alpha=0.7, label='d = 4 (macroscopic)')
        ax.axhline(y=2.0, color='purple', linestyle=':', alpha=0.7, label='d = 2 (Planck scale)')
        ax.axhline(y=D_SPEC_ONE_LOOP, color='gray', linestyle=':', alpha=0.5, label=f'd = 42/11 ≈ {D_SPEC_ONE_LOOP:.3f}')
        
        # Mark crossover scale
        k_cross = find_critical_scale(3.0)
        ax.axvline(x=k_cross, color='orange', linestyle='--', alpha=0.5, label=f'k_cross = {k_cross:.2f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('RG scale k/k_c', fontsize=12)
        ax.set_ylabel(r'Spectral dimension $d_{spec}$', fontsize=12)
        ax.set_title('Spectral Dimension Flow (IRH21.md §2.1, Theorem 2.1)', fontsize=14)
        ax.legend(loc='center right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1.5, 4.5)
        
        # Add annotation
        ax.annotate(
            'UV\n(Planck)',
            xy=(k_range[1]*0.5, 2.1),
            fontsize=10,
            ha='center'
        )
        ax.annotate(
            'IR\n(Macro)',
            xy=(k_range[0]*2, 3.9),
            fontsize=10,
            ha='center'
        )
        
        return fig, ax
    
    def plot_vs_t(
        self,
        t_range: Tuple[float, float] = (-10, 10),
        n_points: int = 200,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot spectral dimension vs RG time t = log(k/k_c).
        
        Parameters
        ----------
        t_range : tuple
            Range of RG time
        n_points : int
            Number of points
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        t = np.linspace(t_range[0], t_range[1], n_points)
        k = np.exp(t)  # k/k_c = e^t
        
        d_spec = spectral_dimension(k)
        
        ax.plot(t, d_spec, 'b-', linewidth=2.5)
        ax.fill_between(t, 2, d_spec, where=d_spec > 2, alpha=0.2, color='blue')
        
        # Mark asymptotes
        ax.axhline(y=4.0, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=2.0, color='purple', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('RG time t = log(k/k_c)', fontsize=12)
        ax.set_ylabel(r'$d_{spec}(t)$', fontsize=12)
        ax.set_title('Spectral Dimension vs RG Time (IRH21.md Eq. 2.8-2.9)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1.5, 4.5)
        
        # Add annotations for limits
        ax.annotate(
            r'$d_{spec} \to 4$ (IR)',
            xy=(t_range[0] + 1, 3.8),
            fontsize=11,
            color='green'
        )
        ax.annotate(
            r'$d_{spec} \to 2$ (UV)',
            xy=(t_range[1] - 3, 2.2),
            fontsize=11,
            color='purple'
        )
        
        return fig, ax
    
    def plot_graviton_contribution(
        self,
        k_range: Tuple[float, float] = (0.01, 100),
        n_points: int = 200,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot graviton correction contribution.
        
        Parameters
        ----------
        k_range : tuple
            Range of RG scales
        n_points : int
            Number of points
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_points)
        delta = graviton_correction(k)
        
        ax.plot(k, delta, 'g-', linewidth=2.5, label=r'$\Delta_{grav}(k)$')
        ax.fill_between(k, 0, delta, alpha=0.2, color='green')
        
        ax.set_xscale('log')
        ax.set_xlabel('RG scale k/k_c', fontsize=12)
        ax.set_ylabel(r'Graviton correction $\Delta_{grav}$', fontsize=12)
        ax.set_title('Graviton Contribution to Spectral Dimension (IRH21.md §2.1.2)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annotate maximum correction in IR
        ax.annotate(
            f'Max correction = {D_SPEC_IR - D_SPEC_ONE_LOOP:.3f}',
            xy=(k_range[0]*3, (D_SPEC_IR - D_SPEC_ONE_LOOP)*0.8),
            fontsize=11
        )
        
        return fig, ax
    
    def create_animation(
        self,
        t_range: Tuple[float, float] = (-5, 5),
        n_frames: int = 100,
        interval: int = 50
    ) -> FuncAnimation:
        """
        Create animated visualization of spectral dimension flow.
        
        Parameters
        ----------
        t_range : tuple
            Range of RG time for animation
        n_frames : int
            Number of animation frames
        interval : int
            Milliseconds between frames
            
        Returns
        -------
        FuncAnimation
            Matplotlib animation object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        t_values = np.linspace(t_range[0], t_range[1], 200)
        k_values = np.exp(t_values)
        d_spec_all = spectral_dimension(k_values)
        
        # Left: d_spec vs t
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        point1, = ax1.plot([], [], 'ro', markersize=10)
        ax1.set_xlim(t_range)
        ax1.set_ylim(1.5, 4.5)
        ax1.axhline(y=4.0, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y=2.0, color='purple', linestyle='--', alpha=0.5)
        ax1.set_xlabel('RG time t')
        ax1.set_ylabel(r'$d_{spec}$')
        ax1.set_title('Spectral Dimension vs RG Time')
        ax1.grid(True, alpha=0.3)
        
        # Right: conceptual dimension visualization
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.set_title('Effective Spacetime Structure')
        
        circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
        ax2.add_patch(circle)
        dim_text = ax2.text(0, -1.5, '', ha='center', fontsize=14)
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            dim_text.set_text('')
            return line1, point1, dim_text
        
        def animate(frame):
            # Update left plot
            idx = int(frame * len(t_values) / n_frames)
            t_current = t_values[:idx+1]
            d_current = d_spec_all[:idx+1]
            line1.set_data(t_current, d_current)
            point1.set_data([t_values[idx]], [d_spec_all[idx]])
            
            # Update dimension text
            d = d_spec_all[idx]
            dim_text.set_text(f'd_spec = {d:.2f}')
            
            # Update circle size to represent effective dimension
            circle.set_radius(0.5 + 0.3 * (d - 2))
            
            return line1, point1, dim_text
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=interval, blit=True
        )
        
        return anim


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def plot_spectral_dimension_flow(**kwargs) -> Tuple[Figure, Axes]:
    """
    Plot spectral dimension vs RG scale.
    
    Theoretical Reference:
        IRH21.md §2.1, Theorem 2.1
    """
    animator = SpectralDimensionAnimator()
    return animator.plot_flow(**kwargs)


def plot_spectral_dimension_vs_scale(**kwargs) -> Tuple[Figure, Axes]:
    """
    Plot spectral dimension vs RG time.
    
    Theoretical Reference:
        IRH21.md §2.1, Eq. 2.8-2.9
    """
    animator = SpectralDimensionAnimator()
    return animator.plot_vs_t(**kwargs)


def create_spectral_animation(**kwargs) -> FuncAnimation:
    """
    Create animated spectral dimension visualization.
    
    Theoretical Reference:
        IRH21.md §2.1
    """
    animator = SpectralDimensionAnimator()
    return animator.create_animation(**kwargs)


# =============================================================================
# Interactive Plotly Version
# =============================================================================

def create_interactive_spectral_plot(
    k_range: Tuple[float, float] = (0.01, 100),
    n_points: int = 200
) -> Any:
    """
    Create interactive spectral dimension plot using Plotly.
    
    Theoretical Reference:
        IRH21.md §2.1
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive plots")
    
    k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_points)
    d_grav = spectral_dimension(k, include_graviton=True)
    d_one_loop = spectral_dimension(k, include_graviton=False)
    
    fig = go.Figure()
    
    # Main trace
    fig.add_trace(go.Scatter(
        x=k, y=d_grav,
        mode='lines',
        name='d_spec with graviton',
        line=dict(color='blue', width=2)
    ))
    
    # One-loop trace
    fig.add_trace(go.Scatter(
        x=k, y=d_one_loop,
        mode='lines',
        name='d_spec one-loop',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Reference lines
    fig.add_hline(y=4.0, line_dash="dot", line_color="green", annotation_text="d=4 (IR)")
    fig.add_hline(y=2.0, line_dash="dot", line_color="purple", annotation_text="d=2 (UV)")
    
    fig.update_layout(
        title='Interactive Spectral Dimension Flow (IRH21.md §2.1)',
        xaxis_title='RG scale k/k_c',
        yaxis_title='d_spec',
        xaxis_type='log',
        yaxis_range=[1.5, 4.5],
        width=900,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    return fig


__all__ = [
    'SpectralDimensionAnimator',
    'plot_spectral_dimension_flow',
    'plot_spectral_dimension_vs_scale',
    'create_spectral_animation',
    'create_interactive_spectral_plot',
    'spectral_dimension',
    'graviton_correction',
    'find_critical_scale',
    'D_SPEC_UV',
    'D_SPEC_IR',
    'D_SPEC_ONE_LOOP',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
]
