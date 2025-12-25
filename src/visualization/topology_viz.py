"""
Topology Visualization Module for IRH v21.0

THEORETICAL FOUNDATION: IRH21.md §3.1, Appendix D

This module provides visualization tools for topological structures:
    - Vortex Wave Pattern (VWP) configurations
    - Instanton charge distributions (n_inst = 3)
    - Betti number visualization (β₁ = 12)
    - Fermion mass hierarchy spectrum

Key Topological Results:
    β₁ = 12 → SU(3)×SU(2)×U(1) gauge group (Appendix D.1)
    n_inst = 3 → Three fermion generations (Appendix D.2)
    K_f values → Mass hierarchy via Yukawa couplings (Eq. 3.6)

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
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any
    Axes = Any
    Axes3D = Any

# Optional plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript Part 1 §3.1, Appendix D"


# =============================================================================
# Physical Constants (from IRH21.md)
# =============================================================================

# Betti number determining gauge group
BETTI_1 = 12  # H₁(M³; ℤ) = ℤ¹²

# Instanton number determining generations
N_INST = 3

# Topological complexity eigenvalues K_f for fermions (Appendix D.3)
# These determine mass hierarchy via m_f ∝ √K_f
FERMION_K_VALUES = {
    # Charged leptons
    'electron': 1.0,
    'muon': 207.0,
    'tau': 3477.0,
    
    # Up-type quarks
    'up': 5.0,
    'charm': 2600.0,
    'top': 340000.0,
    
    # Down-type quarks
    'down': 10.0,
    'strange': 200.0,
    'bottom': 8600.0,
    
    # Neutrinos (very small K_f values)
    'nu_e': 1e-12,
    'nu_mu': 1e-10,
    'nu_tau': 1e-9,
}

# Experimental masses (GeV) for comparison
FERMION_MASSES_EXP = {
    'electron': 0.000511,
    'muon': 0.106,
    'tau': 1.777,  # From experimental measurement (for comparison)
    'up': 0.0022,
    'charm': 1.28,
    'top': 173.0,
    'down': 0.0047,
    'strange': 0.095,
    'bottom': 4.18,
}


# =============================================================================
# Topology Visualizer Class
# =============================================================================

@dataclass
class TopologyVisualizer:
    """
    Visualization system for IRH topological structures.
    
    Theoretical Reference:
        IRH21.md §3.1, Appendix D
        Topological invariants determine gauge group and matter content.
    """
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    colormap: str = 'viridis'
    
    def __post_init__(self):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for TopologyVisualizer")
    
    # Theoretical Reference: IRH v21.4

    
    def plot_betti_numbers(
        self,
        show_decomposition: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Visualize Betti numbers and gauge group emergence.
        
        β₁ = 12 = 8 + 3 + 1 → SU(3) × SU(2) × U(1)
        
        Parameters
        ----------
        show_decomposition : bool
            Show decomposition into gauge group generators
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
        
        # Betti numbers for M³
        betti_numbers = [1, BETTI_1, 0, 1]  # β₀, β₁, β₂, β₃
        x_labels = [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', r'$\beta_3$']
        
        colors = ['gray', 'blue', 'gray', 'gray']
        bars = ax.bar(x_labels, betti_numbers, color=colors, edgecolor='black', linewidth=1.5)
        
        # Highlight β₁ = 12
        bars[1].set_color('royalblue')
        bars[1].set_edgecolor('darkblue')
        bars[1].set_linewidth(2)
        
        # Add value labels
        for bar, val in zip(bars, betti_numbers):
            if val > 0:
                ax.annotate(
                    str(val),
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold'
                )
        
        if show_decomposition:
            # Show decomposition of β₁ = 12
            decomp_text = (
                r'$\beta_1 = 12 = 8 + 3 + 1$'
                '\n'
                r'$\Rightarrow SU(3) \times SU(2) \times U(1)$'
            )
            ax.annotate(
                decomp_text,
                xy=(1, 10),
                fontsize=12,
                ha='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )
            
            # Add mini-bar for decomposition
            inset_ax = ax.inset_axes([0.65, 0.5, 0.3, 0.35])
            gauge_groups = ['SU(3)', 'SU(2)', 'U(1)']
            generators = [8, 3, 1]
            colors_gauge = ['red', 'green', 'blue']
            inset_ax.barh(gauge_groups, generators, color=colors_gauge, alpha=0.7)
            inset_ax.set_xlabel('Generators')
            inset_ax.set_title('Gauge Group', fontsize=10)
        
        ax.set_ylabel('Betti Number', fontsize=12)
        ax.set_title(r'Homology of $M^3 = G_\infty / \Gamma_R$ (IRH21.md App. D.1)', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        return fig, ax
    
    # Theoretical Reference: IRH v21.4

    
    def plot_instanton_number(
        self,
        show_generations: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Visualize instanton number n_inst = 3 → three generations.
        
        Parameters
        ----------
        show_generations : bool
            Show connection to fermion generations
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
        
        # Create visual representation of three instantons
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Three instanton "lumps"
        instanton_positions = [(-2, 0), (0, 0), (2, 0)]
        instanton_charges = [1, 2, 3]
        colors = ['red', 'green', 'blue']
        gen_names = ['1st Gen\n(e, u, d)', '2nd Gen\n(μ, c, s)', '3rd Gen\n(τ, t, b)']
        
        for pos, charge, color, name in zip(instanton_positions, instanton_charges, colors, gen_names):
            # Draw instanton as a filled contour
            x = pos[0] + 0.8 * np.cos(theta)
            y = pos[1] + 0.8 * np.sin(theta)
            ax.fill(x, y, color=color, alpha=0.4)
            ax.plot(x, y, color=color, linewidth=2)
            
            # Mark topological charge
            ax.annotate(
                f'Q = {charge}',
                xy=pos,
                ha='center', va='center',
                fontsize=14, fontweight='bold'
            )
            
            if show_generations:
                ax.annotate(
                    name,
                    xy=(pos[0], pos[1] - 1.3),
                    ha='center', va='top',
                    fontsize=11
                )
        
        # Add arrows connecting them
        for i in range(2):
            ax.annotate(
                '',
                xy=(instanton_positions[i+1][0] - 0.9, 0),
                xytext=(instanton_positions[i][0] + 0.9, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2)
            )
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-2.5, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Title and annotation
        ax.set_title(r'Instanton Number $n_{inst} = 3$ (IRH21.md App. D.2)', fontsize=14)
        ax.text(
            0, 1.8,
            r'$n_{inst} = 3 \Rightarrow$ Three Fermion Generations',
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
        
        return fig, ax
    
    # Theoretical Reference: IRH v21.4

    
    def plot_vwp_configuration(
        self,
        fermion: str = 'electron',
        ax: Optional[Axes3D] = None
    ) -> Tuple[Figure, Axes3D]:
        """
        Visualize Vortex Wave Pattern (VWP) configuration for a fermion.
        
        VWPs are stable topological defects in the cGFT condensate
        that correspond to fermionic degrees of freedom.
        
        Parameters
        ----------
        fermion : str
            Fermion type (electron, muon, tau, up, charm, top, etc.)
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
        
        K_f = FERMION_K_VALUES.get(fermion, 1.0)
        
        # Create VWP visualization as a vortex structure
        t = np.linspace(0, 4*np.pi, 300)
        
        # Complexity determines "winding" of the VWP
        n_windings = int(np.log10(K_f + 1) + 1)
        
        # Parametric vortex
        r = 1 + 0.3 * np.sin(n_windings * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / (4*np.pi) * 2  # Height scales with parameter
        
        # Color by phase
        colors = plt.cm.get_cmap(self.colormap)(t / (4*np.pi))
        
        # Plot as line collection
        for i in range(len(t)-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   color=colors[i], linewidth=2)
        
        # Add core
        ax.scatter([0], [0], [1], s=100, c='red', marker='o', label='VWP core')
        
        # Labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(r'$\phi$-direction')
        ax.set_title(f'{fermion.capitalize()} VWP (K_f = {K_f:.1f})\nIRH21.md App. D.2-D.3', fontsize=14)
        
        # Add K_f annotation
        ax.text2D(0.02, 0.98, f'Topological complexity K_f = {K_f:.1f}',
                 transform=ax.transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return fig, ax
    
    def plot_fermion_spectrum(
        self,
        show_generations: bool = True,
        log_scale: bool = True,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot fermion mass spectrum from topological complexity.
        
        Mass hierarchy: m_f ∝ √K_f (IRH21.md Eq. 3.6)
        
        Parameters
        ----------
        show_generations : bool
            Group by fermion generations
        log_scale : bool
            Use logarithmic scale for masses
        ax : Axes, optional
            Axes to plot on
            
        Returns
        -------
        tuple
            (Figure, Axes) matplotlib objects
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        # Define fermions by generation
        generations = [
            ['electron', 'up', 'down', 'nu_e'],
            ['muon', 'charm', 'strange', 'nu_mu'],
            ['tau', 'top', 'bottom', 'nu_tau']
        ]
        gen_colors = ['blue', 'green', 'red']
        gen_names = ['1st Generation', '2nd Generation', '3rd Generation']
        
        # Compute predicted masses from K_f
        # Normalize to electron mass
        m_e = 0.000511  # GeV
        K_e = FERMION_K_VALUES['electron']
        
        all_fermions = []
        predicted_masses = []
        k_values = []
        exp_masses = []
        gen_indices = []
        
        for gen_idx, gen in enumerate(generations):
            for fermion in gen:
                K_f = FERMION_K_VALUES[fermion]
                # m_f / m_e = sqrt(K_f / K_e)
                m_pred = m_e * np.sqrt(K_f / K_e)
                
                all_fermions.append(fermion)
                k_values.append(K_f)
                predicted_masses.append(m_pred)
                exp_masses.append(FERMION_MASSES_EXP.get(fermion, np.nan))
                gen_indices.append(gen_idx)
        
        x = np.arange(len(all_fermions))
        width = 0.35
        
        # Plot bars
        colors = [gen_colors[g] for g in gen_indices]
        
        bars1 = ax.bar(x - width/2, predicted_masses, width, 
                       label='IRH Prediction', color=colors, alpha=0.7)
        
        # Plot experimental values where available
        exp_vals = [m if not np.isnan(m) else 0 for m in exp_masses]
        bars2 = ax.bar(x + width/2, exp_vals, width,
                       label='Experimental', color='gray', alpha=0.5)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel('Fermion', fontsize=12)
        ax.set_ylabel('Mass (GeV)', fontsize=12)
        ax.set_title('Fermion Mass Spectrum from Topological Complexity\n(IRH21.md §3.2, Eq. 3.6)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(all_fermions, rotation=45, ha='right')
        
        # Add generation legend
        legend_elements = [
            mpatches.Patch(facecolor=gen_colors[i], alpha=0.7, label=gen_names[i])
            for i in range(3)
        ]
        legend_elements.append(mpatches.Patch(facecolor='gray', alpha=0.5, label='Experimental'))
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig, ax
    
    # Theoretical Reference: IRH v21.4

    
    def plot_mass_hierarchy(
        self,
        fermion_type: str = 'leptons',
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot mass hierarchy within a fermion family.
        
        Parameters
        ----------
        fermion_type : str
            'leptons', 'up_quarks', or 'down_quarks'
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
        
        families = {
            'leptons': ['electron', 'muon', 'tau'],
            'up_quarks': ['up', 'charm', 'top'],
            'down_quarks': ['down', 'strange', 'bottom']
        }
        
        fermions = families.get(fermion_type, families['leptons'])
        
        # Get K values
        k_vals = [FERMION_K_VALUES[f] for f in fermions]
        
        # Plot K_f values
        x = [1, 2, 3]
        ax.semilogy(x, k_vals, 'o-', markersize=15, linewidth=2, color='blue')
        
        for xi, k, name in zip(x, k_vals, fermions):
            ax.annotate(
                f'{name}\nK = {k:.0f}',
                xy=(xi, k),
                xytext=(0.2, 0),
                textcoords='offset fontsize',
                ha='left', va='center',
                fontsize=11
            )
        
        ax.set_xticks(x)
        ax.set_xticklabels(['1st Gen', '2nd Gen', '3rd Gen'])
        ax.set_ylabel(r'Topological Complexity $K_f$', fontsize=12)
        ax.set_title(f'{fermion_type.replace("_", " ").title()} Mass Hierarchy\n(IRH21.md Eq. 3.6)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add ratio annotation
        ratio_12 = k_vals[1] / k_vals[0]
        ratio_23 = k_vals[2] / k_vals[1]
        ax.text(
            0.95, 0.05,
            f'K₂/K₁ = {ratio_12:.0f}\nK₃/K₂ = {ratio_23:.0f}',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )
        
        return fig, ax


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Theoretical Reference: IRH v21.4 Part 2, Appendix D.2
def plot_vwp_configuration(fermion: str = 'electron', **kwargs) -> Tuple[Figure, Axes3D]:
    """
    Plot VWP configuration for a fermion.
    
    Theoretical Reference:
        IRH21.md Appendix D.2-D.3
    """
    viz = TopologyVisualizer()
    return viz.plot_vwp_configuration(fermion=fermion, **kwargs)


# Theoretical Reference: IRH v21.4



def plot_instanton_charge(**kwargs) -> Tuple[Figure, Axes]:
    """
    Visualize instanton number n_inst = 3.
    
    Theoretical Reference:
        IRH21.md Appendix D.2
    """
    viz = TopologyVisualizer()
    return viz.plot_instanton_number(**kwargs)

 # Theoretical Reference: IRH v21.4 Part 2, Appendix D.1

def plot_betti_numbers(**kwargs) -> Tuple[Figure, Axes]:
    """
    Visualize Betti numbers and gauge group.
    
    Theoretical Reference:
        IRH21.md Appendix D.1
    """
    viz = TopologyVisualizer()
    return viz.plot_betti_numbers(**kwargs)


def plot_fermion_spectrum(**kwargs) -> Tuple[Figure, Axes]:
    """
    Plot fermion mass spectrum from K_f values.
    
    Theoretical Reference:
        IRH21.md §3.2, Eq. 3.6
    """
    viz = TopologyVisualizer()
    return viz.plot_fermion_spectrum(**kwargs)


# =============================================================================
# Interactive Plotly Version
# =============================================================================

def create_interactive_fermion_spectrum() -> Any:
    """
    Create interactive fermion spectrum using Plotly.
    
    Theoretical Reference:
        IRH21.md §3.2
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for interactive plots")
    
    # Prepare data
    fermions = list(FERMION_K_VALUES.keys())
    k_values = list(FERMION_K_VALUES.values())
    
    # Assign generations
    gen1 = ['electron', 'up', 'down', 'nu_e']
    gen2 = ['muon', 'charm', 'strange', 'nu_mu']
    gen3 = ['tau', 'top', 'bottom', 'nu_tau']
    
    colors = []
    for f in fermions:
        if f in gen1:
            colors.append('blue')
        elif f in gen2:
            colors.append('green')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=fermions,
        y=k_values,
        marker_color=colors,
        text=[f'K={k:.1e}' for k in k_values],
        textposition='outside',
        hovertemplate='%{x}<br>K_f = %{y:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Topological Complexity K_f by Fermion (IRH21.md Eq. 3.6)',
        xaxis_title='Fermion',
        yaxis_title='K_f (log scale)',
        yaxis_type='log',
        width=1000,
        height=600
    )
    
    return fig


__all__ = [
    'TopologyVisualizer',
    'plot_vwp_configuration',
    'plot_instanton_charge',
    'plot_betti_numbers',
    'plot_fermion_spectrum',
    'create_interactive_fermion_spectrum',
    'BETTI_1',
    'N_INST',
    'FERMION_K_VALUES',
    'FERMION_MASSES_EXP',
    'MATPLOTLIB_AVAILABLE',
    'PLOTLY_AVAILABLE',
]
