"""
IRH v21.0 Visualization System

THEORETICAL FOUNDATION: IRH21.md §1-3, docs/ROADMAP.md

This module provides comprehensive visualization capabilities for IRH computations:
    - RG flow phase diagrams and streamlines
    - Group manifold G_inf = SU(2) × U(1)_φ 3D rendering
    - Spectral dimension d_spec(k) flow animations
    - Vortex Wave Pattern (VWP) topological structures

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1-3, docs/ROADMAP.md"

from .rg_flow_plots import (
    RGFlowPlotter,
    plot_phase_diagram_2d,
    plot_phase_diagram_3d,
    plot_rg_trajectory,
    plot_beta_functions,
    plot_fixed_point_stability,
)

from .manifold_viz import (
    ManifoldVisualizer,
    plot_su2_sphere,
    plot_u1_circle,
    plot_g_inf_product,
    plot_geodesics,
)

from .spectral_dimension_viz import (
    SpectralDimensionAnimator,
    plot_spectral_dimension_flow,
    plot_spectral_dimension_vs_scale,
    create_spectral_animation,
)

from .topology_viz import (
    TopologyVisualizer,
    plot_vwp_configuration,
    plot_instanton_charge,
    plot_betti_numbers,
    plot_fermion_spectrum,
)

__all__ = [
    # RG Flow
    'RGFlowPlotter',
    'plot_phase_diagram_2d',
    'plot_phase_diagram_3d',
    'plot_rg_trajectory',
    'plot_beta_functions',
    'plot_fixed_point_stability',
    
    # Manifold
    'ManifoldVisualizer',
    'plot_su2_sphere',
    'plot_u1_circle',
    'plot_g_inf_product',
    'plot_geodesics',
    
    # Spectral Dimension
    'SpectralDimensionAnimator',
    'plot_spectral_dimension_flow',
    'plot_spectral_dimension_vs_scale',
    'create_spectral_animation',
    
    # Topology
    'TopologyVisualizer',
    'plot_vwp_configuration',
    'plot_instanton_charge',
    'plot_betti_numbers',
    'plot_fermion_spectrum',
]
