"""
IRH Desktop - Visualization Package

Provides visualization components for IRH computations:
- RG Flow trajectories
- Spectral dimension flow
- Fixed point visualization
- Group manifold structures

This implements Phase 5 of the DEB_PACKAGE_ROADMAP.md.

Author: Brandon D. McCrary
"""

from irh_desktop.visualization.plots import (
    RGFlowPlot,
    SpectralDimensionPlot,
    FixedPointPlot,
    create_rg_flow_figure,
    create_spectral_dimension_figure,
)

__all__ = [
    "RGFlowPlot",
    "SpectralDimensionPlot",
    "FixedPointPlot",
    "create_rg_flow_figure",
    "create_spectral_dimension_figure",
]
