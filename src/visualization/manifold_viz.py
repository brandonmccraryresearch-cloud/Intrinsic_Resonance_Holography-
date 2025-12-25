"""
Group Manifold Visualization Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §1.1, Appendix A

This module provides visualization tools for the informational group manifold:
    - SU(2) ≅ S³ quaternion space rendering
    - U(1)_φ phase circle visualization
    - G_inf = SU(2) × U(1)_φ product space
    - Geodesic paths and Haar measure sampling

Key Mathematical Objects:
    G_inf = SU(2) × U(1)_φ (Eq. 1.1)
    SU(2) parametrized by quaternions: q = q₀ + iq₁ + jq₂ + kq₃, |q|² = 1
    U(1)_φ: φ ∈ [0, 2π)

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Circle, Wedge, FancyArrowPatch
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import proj3d
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any
    Axes3D = Any

# Optional plotly import
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §1.1, Appendix A"


# =============================================================================
# Quaternion and SU(2) Utilities
# =============================================================================

def quaternion_to_su2(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion q = (q₀, q₁, q₂, q₃) to SU(2) matrix representation.
    
    Theoretical Reference:
        IRH21.md §1.1, Appendix G
        SU(2) matrix: [[q₀ + iq₃, q₂ + iq₁], [-q₂ + iq₁, q₀ - iq₃]]
    """
    q0, q1, q2, q3 = q
    return np.array([
        [q0 + 1j*q3, q2 + 1j*q1],
        [-q2 + 1j*q1, q0 - 1j*q3]
    ])


# Theoretical Reference: IRH v21.4



def su2_to_hopf(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Map quaternion on S³ to point on S² via Hopf fibration.
    
    # Theoretical Reference:
        IRH21.md Appendix G
        Hopf map: S³ → S²
    """
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    x = 2 * (q1*q3 + q0*q2)
    y = 2 * (q2*q3 - q0*q1)
    z = q0**2 - q1**2 - q2**2 + q3**2
    return x, y, z


# Theoretical Reference: IRH v21.4



def sample_su2_haar(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample n_samples quaternions uniformly on S³ (Haar measure on SU(2)).
    
    # Theoretical Reference:
        IRH21.md Appendix A
        Haar measure is the unique left-right invariant measure on SU(2).
    """
    rng = np.random.default_rng(seed)
    # Use the standard method: sample from 4D Gaussian and normalize
    points = rng.standard_normal((n_samples, 4))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


# Theoretical Reference: IRH v21.4



def geodesic_su2(q1: np.ndarray, q2: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute geodesic on S³ from q1 to q2 at parameter values t ∈ [0, 1].
    
    # Theoretical Reference:
        IRH21.md Appendix A (QNCD metric)
        Geodesics on S³ are great circles.
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Handle antipodal points
    if np.dot(q1, q2) < 0:
        q2 = -q2
    
    # Spherical linear interpolation (SLERP)
    dot = np.clip(np.dot(q1, q2), -1, 1)
    theta = np.arccos(dot)
    
    if theta < 1e-10:
        return np.outer(np.ones_like(t), q1)
    
    sin_theta = np.sin(theta)
    return (np.outer(np.sin((1-t)*theta), q1) + np.outer(np.sin(t*theta), q2)) / sin_theta


# =============================================================================
# Manifold Visualizer Class
# =============================================================================

@dataclass
class ManifoldVisualizer:
    """
    Visualization system for the informational group manifold G_inf.
    
    Theoretical Reference:
        IRH21.md §1.1
        G_inf = SU(2) × U(1)_φ is the fundamental symmetry group.
    """
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 100
    colormap: str = 'viridis'
    
    def __post_init__(self):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for ManifoldVisualizer")
    
    # Theoretical Reference: IRH v21.4

    
    def plot_su2_via_hopf(
        self,
        n_samples: int = 1000,
        show_fibers: bool = True,
        ax: Optional[Axes3D] = None
    ) -> Tuple[Figure, Axes3D]:
        """
        Visualize SU(2) ≅ S³ via Hopf fibration projection to S².
        
        The Hopf fibration projects S³ → S² with S¹ fibers.
        
        Parameters
        ----------
        n_samples : int
            Number of sample points
        show_fibers : bool
            Whether to show sample Hopf fibers
        ax : Axes3D, optional
            3D axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes3D) matplotlib objects
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        # Sample SU(2) uniformly
        quaternions = sample_su2_haar(n_samples)
        
        # Project to S² via Hopf map
        projections = np.array([su2_to_hopf(q) for q in quaternions])
        
        # Plot base sphere S²
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')
        
        # Plot projected points with color by original q₀ component
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            projections[:, 2],
            c=quaternions[:, 0],
            cmap=self.colormap,
            s=10,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='q₀ component', shrink=0.6)
        
        if show_fibers:
            # Show a few sample Hopf fibers
            n_fibers = 5
            base_points = projections[:n_fibers]
            for i, bp in enumerate(base_points):
                fiber = self._compute_hopf_fiber(bp)
                ax.plot(fiber[:, 0], fiber[:, 1], fiber[:, 2], 
                       'r-', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('SU(2) via Hopf Fibration (IRH21.md §1.1)\nBase space S², fibers S¹')
        
        return fig, ax
    
    def _compute_hopf_fiber(self, base_point: np.ndarray, n_points: int = 50) -> np.ndarray:
        """Compute a Hopf fiber over a base point on S²."""
        x, y, z = base_point / np.linalg.norm(base_point)
        
        # Inverse Hopf: given (x,y,z) on S², find one preimage
        r = np.sqrt((1 + z) / 2) if z > -0.99 else 0.01
        s = np.sqrt((1 - z) / 2) if z < 0.99 else 0.01
        
        # Generate fiber by rotating in the fiber direction
        theta = np.linspace(0, 2*np.pi, n_points)
        fiber = []
        for t in theta:
            # Quaternion parametrization of fiber
            q0 = r * np.cos(t)
            q1 = s * np.sin(t) if x != 0 else 0
            q2 = s * np.cos(t) if y != 0 else 0
            q3 = r * np.sin(t)
            q = np.array([q0, q1, q2, q3])
            q = q / np.linalg.norm(q)
            # Project back to visualize in 3D (use first 3 components)
            fiber.append([q[1], q[2], q[3]])
        
        return np.array(fiber)
    
    # Theoretical Reference: IRH v21.4

    
    def plot_u1_circle(
        self,
        special_phases: Optional[List[float]] = None,
        show_labels: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Visualize U(1)_φ phase circle.
        
        Parameters
        ----------
        special_phases : list, optional
            List of special phase values to mark
        show_labels : bool
            Whether to show labels
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        # Draw circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
        
        # Mark standard phases
        standard_phases = [0, np.pi/2, np.pi, 3*np.pi/2]
        standard_labels = ['0', 'π/2', 'π', '3π/2']
        
        for phi, label in zip(standard_phases, standard_labels):
            x, y = np.cos(phi), np.sin(phi)
            ax.plot(x, y, 'ko', markersize=8)
            if show_labels:
                ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(1.2*x, 1.2*y),
                    fontsize=12,
                    ha='center', va='center'
                )
        
        # Mark special phases if provided
        if special_phases:
            for phi in special_phases:
                x, y = np.cos(phi), np.sin(phi)
                ax.plot(x, y, 'r*', markersize=15)
        
        # Draw arrow indicating direction
        arrow_angle = np.pi/4
        ax.annotate(
            '',
            xy=(np.cos(arrow_angle + 0.2), np.sin(arrow_angle + 0.2)),
            xytext=(np.cos(arrow_angle), np.sin(arrow_angle)),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2)
        )
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title('U(1)_φ Phase Circle (IRH21.md §1.1)', fontsize=14)
        ax.set_xlabel(r'$\cos(\phi)$')
        ax.set_ylabel(r'$\sin(\phi)$')
        
        return fig, ax
    
    # Theoretical Reference: IRH v21.4

    
    def plot_g_inf_product(
        self,
        n_samples: int = 500,
        ax: Optional[Axes3D] = None
    ) -> Tuple[Figure, Axes3D]:
        """
        Visualize G_inf = SU(2) × U(1)_φ product structure.
        
        Uses stereographic projection of S³ to ℝ³ combined with
        color for U(1)_φ phase.
        
        Parameters
        ----------
        n_samples : int
            Number of sample points
        ax : Axes3D, optional
            3D axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes3D) matplotlib objects
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        # Sample SU(2) and U(1) independently
        su2_samples = sample_su2_haar(n_samples)
        u1_phases = np.random.uniform(0, 2*np.pi, n_samples)
        
        # Stereographic projection of S³ → ℝ³
        # Project from north pole (1, 0, 0, 0)
        stereo = np.zeros((n_samples, 3))
        for i, q in enumerate(su2_samples):
            q0, q1, q2, q3 = q
            if q0 < 0.999:  # Avoid north pole
                denom = 1 - q0
                stereo[i] = [q1/denom, q2/denom, q3/denom]
            else:
                stereo[i] = [0, 0, 0]
        
        # Clip to visible region
        mask = np.linalg.norm(stereo, axis=1) < 5
        stereo = stereo[mask]
        u1_phases = u1_phases[mask]
        
        # Plot with phase as color
        scatter = ax.scatter(
            stereo[:, 0],
            stereo[:, 1],
            stereo[:, 2],
            c=u1_phases,
            cmap='hsv',
            s=20,
            alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label=r'$\phi \in [0, 2\pi)$', shrink=0.6)
        
        ax.set_xlabel('Stereo(q₁)')
        ax.set_ylabel('Stereo(q₂)')
        ax.set_zlabel('Stereo(q₃)')
        ax.set_title(r'$G_\infty = SU(2) \times U(1)_\phi$ (IRH21.md §1.1)', fontsize=14)
        
        return fig, ax
    
    # Theoretical Reference: IRH v21.4

    
    def plot_geodesics(
        self,
        n_geodesics: int = 10,
        n_points: int = 50,
        ax: Optional[Axes3D] = None
    ) -> Tuple[Figure, Axes3D]:
        """
        Visualize geodesics (great circles) on S³ projected to ℝ³.
        
        Parameters
        ----------
        n_geodesics : int
            Number of geodesics to plot
        n_points : int
            Points per geodesic
        ax : Axes3D, optional
            3D axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes3D) matplotlib objects
        """
        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        # Sample endpoint pairs
        endpoints = sample_su2_haar(2 * n_geodesics)
        colors = plt.cm.get_cmap(self.colormap)(np.linspace(0, 1, n_geodesics))
        
        t = np.linspace(0, 1, n_points)
        
        for i in range(n_geodesics):
            q1 = endpoints[2*i]
            q2 = endpoints[2*i + 1]
            
            geodesic = geodesic_su2(q1, q2, t)
            
            # Stereographic projection
            stereo = np.zeros((n_points, 3))
            for j, q in enumerate(geodesic):
                q0, q1_, q2_, q3_ = q
                if q0 < 0.999:
                    denom = 1 - q0
                    stereo[j] = [q1_/denom, q2_/denom, q3_/denom]
            
            # Plot geodesic
            ax.plot(
                stereo[:, 0],
                stereo[:, 1],
                stereo[:, 2],
                color=colors[i],
                linewidth=2,
                alpha=0.8
            )
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Geodesics on S³ (QNCD metric, IRH21.md App. A)', fontsize=14)
        
        return fig, ax


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def plot_su2_sphere(**kwargs) -> Tuple[Figure, Axes3D]:
    """
    Visualize SU(2) ≅ S³ via Hopf fibration.
    
    Theoretical Reference:
        IRH21.md §1.1
    """
    viz = ManifoldVisualizer()
    return viz.plot_su2_via_hopf(**kwargs)


def plot_u1_circle(**kwargs) -> Tuple[Figure, Axes]:
    """
    Visualize U(1)_φ phase circle.
    
    Theoretical Reference:
        IRH21.md §1.1
    """
    viz = ManifoldVisualizer()
    return viz.plot_u1_circle(**kwargs)


def plot_g_inf_product(**kwargs) -> Tuple[Figure, Axes3D]:
    """
    Visualize G_inf = SU(2) × U(1)_φ.
    
    # Theoretical Reference:
        IRH21.md §1.1, Eq. 1.1
    """
    viz = ManifoldVisualizer()
    return viz.plot_g_inf_product(**kwargs)


# Theoretical Reference: IRH v21.4 Part 1, §2.2
def plot_geodesics(**kwargs) -> Tuple[Figure, Axes3D]:
    """
    Visualize geodesics on S³.
    
    Theoretical Reference:
        IRH21.md Appendix A (QNCD metric)
    """
    viz = ManifoldVisualizer()
    return viz.plot_geodesics(**kwargs)


__all__ = [
    'ManifoldVisualizer',
    'plot_su2_sphere',
    'plot_u1_circle',
    'plot_g_inf_product',
    'plot_geodesics',
    'quaternion_to_su2',
    'su2_to_hopf',
    'sample_su2_haar',
    'geodesic_su2',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
]
