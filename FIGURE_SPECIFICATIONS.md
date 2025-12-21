# Figure Specifications for IRH v21.4 LaTeX Document

## Overview
This document provides detailed specifications for all figures to be created and inserted into the LaTeX version of the Intrinsic Resonance Holography v21.4 manuscript. Each figure placeholder in the LaTeX document should be replaced with a visualization conforming to these specifications.

---

## Figure 2.1: Renormalization Group Flow of the Spectral Dimension

### Location
Section 2.1: The Infrared Geometry and the Exact Emergence of 4D Spacetime

### Figure Type
Line plot with shaded uncertainty region

### Detailed Specifications

#### Axes
- **X-axis**: Energy Scale $k$ (in units of $\ell_0^{-1}$)
  - Scale: Logarithmic
  - Range: From $10^{-5}$ (deep IR) to $10^1$ (UV)
  - Label: "Energy Scale $k$ (in units of $\ell_0^{-1}$, logarithmic scale)"
  - Grid: Major gridlines at powers of 10, minor gridlines at intermediate values

- **Y-axis**: Spectral Dimension $d_{\text{spec}}(k)$
  - Scale: Linear
  - Range: From 1.5 to 4.5
  - Label: "Spectral Dimension $d_{\text{spec}}(k)$"
  - Grid: Major gridlines at integer values (2, 3, 4), minor at 0.5 intervals

#### Main Curve
- **Line Color**: Dark blue (#003366)
- **Line Width**: 2.5 points
- **Line Style**: Solid

#### Key Points to Mark
1. **UV Region** ($k \to \ell_0^{-1}$, $k \approx 1$):
   - $d_{\text{spec}} \approx 2.0$
   - Mark with red circle
   - Annotation: "UV: $d_{\text{spec}} \approx 2$ (dimensional reduction)"

2. **Intermediate Region** ($k \approx 10^{-2}$ to $10^{-3}$):
   - $d_{\text{spec}} \approx 3.818$ (42/11)
   - Mark with orange diamond
   - Annotation: "One-loop fixed point: $d_{\text{spec}} = 42/11 \approx 3.818$"

3. **Deep IR** ($k \to 0$, $k < 10^{-4}$):
   - $d_{\text{spec}} \to 4.000$ (exactly)
   - Mark with green square
   - Annotation: "IR: $d_{\text{spec}} \to 4$ exactly"
   - Horizontal dashed line at $d_{\text{spec}} = 4$ for reference

#### Uncertainty Band
- **Shaded Region**: Around main curve
- **Color**: Light blue with 30% opacity (#CCE5FF, alpha=0.3)
- **Width**: ±0.05 units around the main curve
- **Note**: Represents rigorously bounded theoretical uncertainty from HarmonyOptimizer error propagation

#### Mathematical Annotations
- Add equation box showing: $\frac{\partial d_{\text{spec}}}{\partial t} = -\Delta_{\text{grav}}(k)$
- Position: Top right corner
- Background: Semi-transparent white box

#### Data Points
- Show 10^14 computational verification points as:
  - Tiny dots in gray (#808080)
  - Density: Higher in transition regions
  - Size: 0.5 points
  - Opacity: 50%

#### Legend
Position: Upper left corner
Entries:
1. "RG Flow (HarmonyOptimizer)" - Blue solid line
2. "Uncertainty Band (±0.05)" - Light blue shading
3. "Computational Points (10¹⁴)" - Gray dots
4. "Theoretical Limit ($d=4$)" - Green dashed line

#### Caption
"Renormalization Group Flow of the Spectral Dimension ($d_{\text{spec}}(k)$) in Intrinsic Resonance Holography. This plot, computationally verified with $10^{14}$ integration points, demonstrates the flow from $d_{\text{spec}} \approx 2$ in the UV, through the one-loop fixed point at $42/11 \approx 3.818$, to exactly 4 in the deep IR due to graviton fluctuations $\Delta_{\text{grav}}(k)$. The shaded region represents rigorously bounded theoretical uncertainty."

---

## Additional Figures (Placeholders Only)

### Figure 3.1: Gauge Group Emergence from First Betti Number
**Location**: Section 3.1.1

**Type**: Schematic diagram

**Elements**:
- Central circle labeled "$\beta_1 = 12$"
- Three emerging branches showing decomposition:
  - Upper: "SU(3): 8 generators" (red)
  - Middle: "SU(2): 3 generators" (blue)
  - Lower: "U(1): 1 generator" (green)
- Arrows showing decomposition: $12 = 8 + 3 + 1$
- Mathematical annotation: $\mathrm{SU}(3) \times \mathrm{SU}(2) \times \mathrm{U}(1)$

**Style**: Clean, modern diagram with color-coded sectors

---

### Figure 3.2: Fermion Mass Hierarchy
**Location**: Section 3.2.4

**Type**: Logarithmic bar chart

**Elements**:
- Y-axis: Fermion mass (MeV/c²), logarithmic scale
- X-axis: Fermion species (3 generations × 4 fermion types)
- Bars grouped by generation (1st, 2nd, 3rd)
- Color coding:
  - Charged leptons: Purple
  - Up-type quarks: Red
  - Down-type quarks: Blue
  - Neutrinos: Yellow
- Each bar shows:
  - Height: Mass value
  - Error bar: Theoretical uncertainty
  - Label: Topological complexity $\mathcal{K}_f$

**Data Points** (approximate, to be verified from Table 3.1):
- Electron: ~0.5 MeV, $\mathcal{K}_e = 1.000$
- Muon: ~106 MeV, $\mathcal{K}_\mu = 206.77$
- Tau: ~1777 MeV, $\mathcal{K}_\tau = 3477.4$
- Up quark: ~2.3 MeV
- Charm quark: ~1275 MeV
- Top quark: ~173 GeV
- Down quark: ~4.8 MeV
- Strange quark: ~95 MeV
- Bottom quark: ~4.18 GeV

---

### Figure A.1: QNCD Metric Construction
**Location**: Appendix A

**Type**: Flowchart/Schematic

**Elements**:
1. **Input**: Group elements $g_1, g_2 \in G_{\text{inf}}$
2. **Step 1**: Map to qubit strings
   - Box: "Cayley-Klein Parameterization"
   - Output: Quantum states $|\psi_1\rangle$, $|\psi_2\rangle$
3. **Step 2**: Quantum compression
   - Box: "Quantum Lempel-Ziv Compressor"
   - Output: Compressed lengths $K_Q(|\psi_i\rangle)$
4. **Step 3**: QNCD calculation
   - Formula: $d_{\text{QNCD}}(g_1, g_2) = \frac{K_Q(|\psi_1\rangle|\psi_2\rangle) - \min(K_Q(|\psi_1\rangle), K_Q(|\psi_2\rangle))}{\max(K_Q(|\psi_1\rangle), K_Q(|\psi_2\rangle))}$
5. **Output**: Normalized distance in [0, 1]

**Style**: Clean flowchart with blue boxes and black arrows

---

### Figure C.1: Graviton Propagator Spectral Decomposition
**Location**: Appendix C.2

**Type**: 3D surface plot

**Elements**:
- X-axis: Momentum $p$ (logarithmic, 0.01 to 10 in Planck units)
- Y-axis: Energy scale $k$ (logarithmic, 0.01 to 10)
- Z-axis: Propagator amplitude $G(p,k)$
- Color map: Blue (low) to red (high)
- Contour lines projected onto base plane
- Marked pole at $d_{\text{spec}} = 4$
- Annotation showing $\Delta_{\text{grav}}(k)$ extraction

---

### Figure E.1: Fine-Structure Constant Derivation
**Location**: Section 3.2.2

**Type**: Computational flow diagram

**Elements**:
1. **Input Layer**: 
   - Fixed point couplings $(\lambda_*, \gamma_*, \mu_*)$
   - Topological invariants $(\beta_1, n_{\text{inst}})$
2. **Intermediate Calculations**:
   - Box 1: "Frustration Density $\rho_{\text{frust}}$"
   - Box 2: "Holographic Entropy Bound"
   - Box 3: "UV/IR Matching Condition"
3. **Output**:
   - Large box: "$\alpha^{-1} = 137.035999084(1)$"
   - Comparison arrow to experimental value
4. **Verification**:
   - Green checkmark: "12 decimal place agreement"

**Style**: Hierarchical flowchart with color-coded steps

---

### Figure I.1: Born Rule Emergence
**Location**: Appendix I.2.1

**Type**: Conceptual diagram with mathematical overlay

**Elements**:
- **Left Panel**: Phase History Space $\mathcal{P}$
  - Abstract representation as high-dimensional manifold
  - QNCD metric structure shown as curved grid
  - Regions colored by algorithmic complexity
  
- **Middle Panel**: Coarse-Graining Process
  - Arrows showing mapping from microstates to macrostates
  - Algorithmic Selection mechanism highlighted
  - Pointer basis states marked
  
- **Right Panel**: Probability Distribution
  - Bar chart showing $P_i = |c_i|^2$
  - Correspondence arrows to phase history volumes
  - Mathematical formula: $\mu_{\text{QNCD}}(\mathcal{P}_i) = |c_i|^2$

**Caption**: "Derivation of the Born Rule from statistical mechanics of phase histories. The measure $\mu_{\text{QNCD}}$ on phase space trajectories, weighted by algorithmic complexity, naturally yields the quantum probability amplitudes."

---

### Figure J.1: Generation-Specific LIV Predictions
**Location**: Appendix J.1

**Type**: Multi-panel comparison plot

**Elements**:
- Three panels (electron, muon, tau)
- Each panel shows modified dispersion relation:
  - X-axis: Energy $E$ (GeV, logarithmic)
  - Y-axis: Dispersion deviation $\Delta E / E$ (%)
  - Blue line: Standard dispersion ($E^2 = p^2c^2 + m^2c^4$)
  - Red line: IRH prediction with LIV correction
  - Shaded region: Detectable range for current experiments
- Annotations showing $\xi_f$ values for each generation
- Comparison bar showing relative magnitudes: $\xi_\tau > \xi_\mu > \xi_e$

---

### Figure J.2: Gravitational Wave Sidebands
**Location**: Appendix J.2

**Type**: Frequency spectrum

**Elements**:
- X-axis: Frequency (Hz, logarithmic, 10 to 1000 Hz)
- Y-axis: Strain amplitude (dimensionless, logarithmic, 10^-24 to 10^-20)
- **Main signal**: Standard binary black hole merger chirp (blue thick line)
- **IRH prediction**: Additional sideband peaks (red thin lines)
  - Primary sideband: ±Δf from main frequency
  - Secondary sidebands: ±2Δf, ±3Δf
  - Amplitude: 1/10 to 1/100 of main signal
- **Annotations**:
  - Sideband spacing formula: $\Delta f = \sqrt{\lambda_*/\gamma_*} f_{\text{Planck}}$
  - Sensitivity curves for LIGO, LISA, Einstein Telescope
  - Shaded "detectable" regions

---

## Figure Creation Guidelines

### General Specifications
- **Resolution**: Minimum 300 DPI for all figures
- **Format**: PDF (vector) preferred, PNG (raster) acceptable for complex 3D plots
- **Fonts**: Use Noto Sans for all text in figures to match document
- **Math Fonts**: Use Fira Math or matching math font for equations in figures
- **Color Palette**: Use colorblind-friendly palettes
- **Line Widths**: Minimum 1.5 points for visibility
- **Markers**: Minimum 4 point size
- **Legend**: Clear and positioned to not obscure data

### Software Recommendations
- **Python**: matplotlib, seaborn, plotly
- **Mathematica**: For analytical plots
- **TikZ/PGFPlots**: For LaTeX-native vector graphics
- **Inkscape**: For manual diagram creation
- **MATLAB**: For numerical simulations

### Verification Checklist
For each figure:
- [ ] Axes labeled with units
- [ ] Legend included (if multiple data series)
- [ ] Caption accurately describes content
- [ ] All text legible at final print size
- [ ] Color scheme accessible (colorblind-friendly)
- [ ] Mathematical notation consistent with manuscript
- [ ] Source data/code documented
- [ ] High resolution (≥300 DPI)
- [ ] Proper file format (PDF vector preferred)

---

## Data Availability

All numerical data for figure generation should come from:
1. **HarmonyOptimizer outputs**: Available in repository data/ directory
2. **Analytical formulas**: Specified in manuscript equations
3. **Computational verification**: See certified numerical results in appendices

## Contact

For questions about figure specifications or to submit completed figures:
- Open an issue on the GitHub repository
- Tag with `figure-creation` label
- Include figure number in issue title

---

**Last Updated**: December 2025
**Document Version**: 1.0
