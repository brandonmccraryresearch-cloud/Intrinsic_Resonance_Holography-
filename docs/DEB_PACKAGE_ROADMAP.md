# IRH Desktop Application: .deb Package Implementation Roadmap

**Project**: IRH v21.0 Desktop Interface  
**Document Version**: 1.1  
**Target Platform**: Debian/Ubuntu Linux  
**Status**: ✅ **Phase VI Complete** (December 2025)  
**Last Updated**: December 16, 2025

---

## Executive Summary

This document outlines the implementation roadmap for creating a Debian package (`.deb`) that provides a feature-rich desktop application interface for the Intrinsic Resonance Holography (IRH) v21.0 computational framework. The application will serve as both an engine installation manager and an interactive interface for running IRH computations with maximum transparency and customization.

---

## Table of Contents

1. [Project Goals](#1-project-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [GUI Framework Selection](#3-gui-framework-selection)
4. [Core Features](#4-core-features)
5. [Implementation Phases](#5-implementation-phases)
6. [Technical Specifications](#6-technical-specifications)
7. [Build and Packaging](#7-build-and-packaging)
8. [Testing Strategy](#8-testing-strategy)
9. [Distribution Plan](#9-distribution-plan)
10. [Timeline and Milestones](#10-timeline-and-milestones)

---

## 1. Project Goals

### 1.1 Primary Objectives

1. **Simplified Installation**: One-click installation of IRH computational engine on Debian-based systems
2. **Auto-Update System**: Automatic download and installation of latest IRH repository versions
3. **Transparent Interface**: Verbose, understandable output showing exactly what's happening and why
4. **Customization-Friendly**: Easy modification of parameters, configurations, and computations
5. **Feature-Rich GUI**: Professional desktop interface for scientific computation

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Transparency** | All operations explained in plain language with theoretical references |
| **Modularity** | Components can be updated/replaced independently |
| **Accessibility** | Usable by both experts and newcomers |
| **Reproducibility** | All computations fully reproducible with logged parameters |
| **Extensibility** | Plugin architecture for custom modules |

### 1.3 Target Users

- Theoretical physicists researching unified theories
- Computational scientists working on GFT/asymptotic safety
- Graduate students learning quantum gravity
- Researchers validating IRH predictions
- Educators demonstrating emergent physics concepts

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        IRH Desktop Application                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      GUI Layer (PyQt6/GTK4)                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│  │  │ Main     │ │ Config   │ │ Output   │ │ Viz      │           │   │
│  │  │ Window   │ │ Panel    │ │ Console  │ │ Canvas   │           │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Transparency Engine                           │   │
│  │  - Verbose logging with theoretical context                      │   │
│  │  - Real-time progress with equation references                   │   │
│  │  - Step-by-step explanation of computations                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Core Services Layer                           │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ Engine       │ │ Update       │ │ Config       │             │   │
│  │  │ Manager      │ │ Service      │ │ Manager      │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │   │
│  │  │ Job          │ │ Export       │ │ Plugin       │             │   │
│  │  │ Scheduler    │ │ Service      │ │ System       │             │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    IRH Engine (Python Package)                   │   │
│  │  src/primitives → src/cgft → src/rg_flow → ... → src/observables │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  GitHub Repository │
                    │  (Latest Source)   │
                    └───────────────────┘
```

### 2.2 Component Description

| Component | Purpose |
|-----------|---------|
| **GUI Layer** | User interface for all interactions |
| **Transparency Engine** | Generates verbose, contextual output |
| **Engine Manager** | Installs, updates, manages IRH core |
| **Update Service** | Checks and applies updates from GitHub |
| **Config Manager** | Handles all configuration files |
| **Job Scheduler** | Manages computation queue |
| **Export Service** | Saves results in various formats |
| **Plugin System** | Extends functionality via plugins |

### 2.3 Data Flow

```
User Input → GUI Validation → Transparency Context → Engine Execution → Output Formatting → Display + Log
                                     │                      │
                                     ▼                      ▼
                              Log theoretical         Save computation
                              references              metadata
```

---

## 3. GUI Framework Selection

### 3.1 Framework Comparison

| Framework | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **PyQt6** | Mature, feature-rich, excellent theming | GPL/commercial licensing | ✓ Recommended |
| **GTK4** | Native Linux look, LGPL | Less documentation | Alternative |
| **Kivy** | Cross-platform, mobile-ready | Non-native look | Not recommended |
| **Dear ImGui** | Fast, immediate mode | Limited widgets | Special use |

### 3.2 Recommended: PyQt6

**Rationale**:
- Professional appearance suitable for scientific applications
- Extensive widget library (plots, trees, tables)
- Strong theming support for customization
- Active community and documentation
- Qt Designer for rapid UI development

### 3.3 UI Design Guidelines

```
┌─────────────────────────────────────────────────────────────────────────┐
│  IRH Desktop v21.0                                    [−] [□] [×]       │
├─────────────────────────────────────────────────────────────────────────┤
│  File  Edit  View  Compute  Tools  Help                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────┐ ┌───────────────────────────────────────────┐ │
│ │ Module Navigator      │ │ Workspace                                  │ │
│ │ ─────────────────────│ │ ┌─────────────────────────────────────────┐ │ │
│ │ ▼ Primitives         │ │ │ Current Computation: RG Flow Analysis   │ │ │
│ │   ├─ Quaternions     │ │ │                                         │ │ │
│ │   ├─ Group Manifolds │ │ │ Progress: ████████████░░░░ 75%          │ │ │
│ │   └─ QNCD            │ │ │                                         │ │ │
│ │ ▼ cGFT               │ │ │ Status: Computing β-functions (Eq.1.13) │ │ │
│ │   └─ Actions         │ │ └─────────────────────────────────────────┘ │ │
│ │ ▶ RG Flow            │ │ ┌─────────────────────────────────────────┐ │ │
│ │ ▶ Emergent Physics   │ │ │ Transparent Output Console               │ │ │
│ │ ▶ Predictions        │ │ │ ─────────────────────────────────────────│ │ │
│ │                      │ │ │ [INFO] Starting RG flow computation      │ │ │
│ │ ─────────────────────│ │ │        Theoretical basis: IRH21.md §1.2  │ │ │
│ │ Quick Actions:       │ │ │                                         │ │ │
│ │ [Compute α⁻¹]        │ │ │ [STEP] Evaluating β_λ at current point  │ │ │
│ │ [Verify Fixed Point] │ │ │        Formula: β_λ = -2λ̃ + (9/8π²)λ̃²  │ │ │
│ │ [Run Full Suite]     │ │ │        Current λ̃ = 52.637...            │ │ │
│ │                      │ │ │        Result: β_λ = 2.3×10⁻¹²          │ │ │
│ └───────────────────────┘ │ │        ✓ Near fixed point (|β| < 10⁻¹⁰) │ │ │
│                           │ │                                         │ │ │
│                           │ │ [WHY] The β-function measures how the  │ │ │
│                           │ │       coupling λ̃ changes with energy    │ │ │
│                           │ │       scale. At the Cosmic Fixed Point, │ │ │
│                           │ │       β_λ → 0, meaning λ̃ stops running. │ │ │
│                           │ └─────────────────────────────────────────┘ │ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Features

### 4.1 Engine Management

#### 4.1.1 Initial Installation
```
┌─────────────────────────────────────────────────────────────────┐
│                   IRH Engine Installation                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Welcome to IRH v21.0 Desktop                                   │
│                                                                  │
│  This wizard will install the IRH computational engine.         │
│                                                                  │
│  Installation Options:                                           │
│  ○ Bundled Engine (v21.0.0) - Install included version          │
│  ● Latest from GitHub - Download latest repository              │
│  ○ Custom Location - Use existing installation                  │
│                                                                  │
│  Installation Directory: [/opt/irh/engine    ] [Browse...]      │
│                                                                  │
│  ☑ Create desktop shortcut                                      │
│  ☑ Add to system PATH                                           │
│  ☑ Install Jupyter integration                                  │
│  ☑ Enable automatic updates                                     │
│                                                                  │
│                              [Back] [Next >] [Cancel]           │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 Auto-Update System

**Update Check Flow**:
```python
class UpdateService:
    """
    Automatic update service for IRH engine.
    
    Features:
        - Background update checks
        - Version comparison
        - Incremental updates (git pull)
        - Full reinstall option
        - Rollback capability
    """
    
    def check_for_updates(self) -> UpdateInfo:
        """Check GitHub for newer versions."""
        # Compare local vs remote commit hash
        # Parse version tags
        # Return update info with changelog
        
    def download_update(self, version: str) -> bool:
        """Download update with progress callback."""
        # Git clone or download tarball
        # Verify checksums
        # Extract to staging area
        
    def apply_update(self, backup: bool = True) -> bool:
        """Apply downloaded update."""
        # Backup current installation
        # Replace with new version
        # Run post-install hooks
        # Verify installation
```

**Update UI**:
```
┌─────────────────────────────────────────────────────────────────┐
│                      Update Available                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Current Version: 21.0.0                                        │
│  Available Version: 21.0.1                                      │
│                                                                  │
│  Changes:                                                        │
│  • Fixed eigenvalue computation in stability analysis            │
│  • Added new LIV prediction module                               │
│  • Improved QNCD metric performance by 40%                       │
│                                                                  │
│  ☑ Create backup before updating                                │
│                                                                  │
│  [View Full Changelog] [Update Now] [Remind Later] [Skip]       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Transparent Output System

The transparency engine is central to IRH's philosophy. Every computation must explain itself.

#### 4.2.1 Output Levels

| Level | Purpose | Example |
|-------|---------|---------|
| **INFO** | High-level progress | "Starting RG flow computation" |
| **STEP** | Individual operations | "Computing β_λ at current point" |
| **DETAIL** | Numerical specifics | "λ̃ = 52.637..., result = 2.3×10⁻¹²" |
| **WHY** | Plain-language explanation | "The β-function measures how..." |
| **REF** | Theoretical reference | "IRH21.md §1.2, Eq. 1.13" |
| **WARN** | Potential issues | "Large lattice spacing may affect precision" |
| **ERROR** | Failures with context | "Convergence failed after 1000 iterations" |

#### 4.2.2 Transparency Message Format

```python
@dataclass
class TransparentMessage:
    """
    A message in the transparency system.
    """
    level: str              # INFO, STEP, DETAIL, WHY, REF, WARN, ERROR
    timestamp: datetime
    message: str            # Main message text
    equation: str = None    # LaTeX equation if applicable
    reference: str = None   # IRH21.md reference
    explanation: str = None # Plain-language explanation
    values: dict = None     # Numerical values involved
    
    def render_console(self) -> str:
        """Render for console output."""
        
    def render_gui(self) -> QWidget:
        """Render for GUI display."""
        
    def render_log(self) -> str:
        """Render for log file."""
```

#### 4.2.3 Example Output

```
╔══════════════════════════════════════════════════════════════════════════╗
║ IRH Computation Log - Cosmic Fixed Point Verification                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║ [12:34:56] INFO  Starting fixed point verification                       ║
║           REF   IRH21.md §1.2-1.3                                        ║
║                                                                          ║
║ [12:34:56] STEP  Loading fixed-point coupling values                     ║
║           DETAIL λ̃* = 48π²/9 = 52.63789013914325...                      ║
║           DETAIL γ̃* = 32π²/3 = 105.2757802782865...                      ║
║           DETAIL μ̃* = 16π²   = 157.9136704174297...                      ║
║           REF   Eq. 1.14                                                 ║
║                                                                          ║
║ [12:34:57] STEP  Computing β_λ at fixed point                            ║
║           DETAIL β_λ = -2λ̃ + (9/8π²)λ̃²                                   ║
║           DETAIL    = -2(52.638) + (9/8π²)(52.638)²                      ║
║           DETAIL    = -105.276 + 105.276                                 ║
║           DETAIL    = 2.84×10⁻¹⁴ (numerical precision limit)             ║
║           ✓ PASS  |β_λ| < 10⁻¹⁰ → Fixed point condition satisfied        ║
║                                                                          ║
║ [12:34:57] WHY   At the Cosmic Fixed Point, all three β-functions        ║
║                  vanish simultaneously. This means the coupling          ║
║                  constants stop "running" with energy scale - they       ║
║                  have reached their final, infrared values. This is      ║
║                  the unique attractor of the theory from which all       ║
║                  physical constants emerge.                              ║
║                                                                          ║
║ [12:34:58] INFO  Computing universal exponent C_H                        ║
║           DETAIL C_H = 3λ̃*/(2γ̃*) = 3(52.638)/(2×105.276)                 ║
║           DETAIL    = 0.045935703598...                                  ║
║           REF   Eq. 1.16                                                 ║
║           ✓ MATCH 12-digit agreement with certified value                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 4.3 Customization Features

#### 4.3.1 Configuration Editor

```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration Editor                                             │
├─────────────────────────────────────────────────────────────────┤
│ Profile: [Default ▼] [New] [Save] [Reset]                       │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Category          │ Setting              │ Value             │ │
│ ├───────────────────┼──────────────────────┼───────────────────┤ │
│ │ Lattice           │ N_SU2                │ [50    ]          │ │
│ │                   │ N_U1                 │ [25    ]          │ │
│ │                   │ Spacing              │ [0.02  ]          │ │
│ │ RG Flow           │ Method               │ [RK4 ▼]           │ │
│ │                   │ dt                   │ [0.001 ]          │ │
│ │                   │ t_UV                 │ [10.0  ]          │ │
│ │                   │ t_IR                 │ [-20.0 ]          │ │
│ │ Precision         │ Float dtype          │ [float64 ▼]       │ │
│ │                   │ Tolerance            │ [1e-12 ]          │ │
│ │ Output            │ Verbosity            │ [●●●○○]           │ │
│ │                   │ Show equations       │ [✓]               │ │
│ │                   │ Show explanations    │ [✓]               │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ [Help: What do these settings mean?]                            │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.3.2 Custom Module Interface

Users can create custom computation modules:

```python
# custom_modules/my_analysis.py
from irh.desktop.plugin import IRHPlugin, register_plugin

@register_plugin
class MyCustomAnalysis(IRHPlugin):
    """
    Custom analysis module for IRH Desktop.
    
    This module will appear in the GUI under Tools > Custom Modules.
    """
    
    name = "My Custom Analysis"
    version = "1.0.0"
    author = "User Name"
    description = "Performs custom analysis on RG flow data"
    
    # GUI configuration
    parameters = {
        "n_iterations": {"type": "int", "default": 100, "min": 1},
        "threshold": {"type": "float", "default": 1e-6},
    }
    
    def run(self, context: IRHContext, params: dict) -> IRHResult:
        """Execute the custom analysis."""
        context.log_info("Starting custom analysis...")
        # ... computation logic ...
        return IRHResult(data=results, success=True)
```

### 4.4 Visualization Components

#### 4.4.1 Available Plots

| Plot Type | Purpose | Interactive |
|-----------|---------|-------------|
| RG Flow Trajectory | Show coupling evolution | Yes - zoom, pan |
| Spectral Dimension | d_spec(k) flow to 4 | Yes - scale |
| Fixed Point Basin | Attractiveness visualization | Yes - rotate |
| Group Manifold | SU(2)×U(1) structure | Yes - 3D rotate |
| Convergence Plot | Numerical verification | Yes - zoom |

#### 4.4.2 Plot Widget

```python
class RGFlowPlot(IRHPlotWidget):
    """
    Interactive RG flow visualization.
    
    Features:
        - Real-time trajectory plotting
        - Fixed point marking
        - Phase portrait overlay
        - Export to PDF/PNG/SVG
    """
    
    def plot_trajectory(self, trajectory: RGTrajectory):
        """Plot a single RG trajectory."""
        
    def mark_fixed_point(self, fp: FixedPoint, style: str = "star"):
        """Mark fixed point location."""
        
    def show_phase_portrait(self, enabled: bool = True):
        """Toggle phase portrait background."""
        
    def add_annotation(self, point, text, reference=None):
        """Add theoretical annotation to plot."""
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-4) ✅ COMPLETE

**Deliverables**:
- [x] Project structure and build system
- [x] Core PyQt6 application shell
- [x] Basic window layout and navigation
- [x] Configuration management system
- [x] Logging infrastructure

**Technical Tasks**:
```
irh-desktop/
├── setup.py
├── pyproject.toml
├── src/
│   └── irh_desktop/
│       ├── __init__.py
│       ├── main.py           # Application entry point
│       ├── app.py            # QApplication setup
│       ├── config/           # Configuration management
│       ├── ui/               # UI components
│       │   ├── main_window.py
│       │   ├── module_navigator.py
│       │   └── workspace.py
│       └── core/             # Core services
│           ├── engine_manager.py
│           └── config_manager.py
└── resources/
    ├── icons/
    ├── themes/
    └── translations/
```

### Phase 2: Engine Integration (Weeks 5-8) ✅ COMPLETE

**Deliverables**:
- [x] IRH engine discovery and loading
- [x] Update service implementation
- [x] GitHub integration for latest downloads
- [x] Installation wizard
- [x] Rollback capability

**Key Components**:
```python
class EngineManager:
    """Manages IRH engine lifecycle."""
    
    def discover_engines(self) -> list[EngineInfo]
    def install_engine(self, source: str, path: str) -> bool
    def update_engine(self, engine: EngineInfo) -> bool
    def verify_engine(self, engine: EngineInfo) -> VerificationResult
    def rollback_engine(self, engine: EngineInfo) -> bool
```

### Phase 3: Transparency Engine (Weeks 9-12) ✅ COMPLETE

**Deliverables**:
- [x] Message formatting system
- [x] Real-time output console
- [x] Equation rendering (LaTeX)
- [x] Explanation database
- [x] Log file management

**Output Console Features**:
- Syntax highlighting for different message types
- Collapsible detailed sections
- Search and filter
- Export to file
- Copy with formatting

### Phase 4: Computation Interface (Weeks 13-16) ✅ COMPLETE

**Deliverables**:
- [x] Module browser and launcher (ComputationRunner)
- [x] Parameter input forms (ComputationParameters)
- [x] Progress tracking (progress callbacks)
- [x] Job queue management (JobQueueManager)
- [x] Result display and export (ResultExporter)

**Computation Workflow**:
```
Select Module → Configure Parameters → Start Computation → Monitor Progress → View Results → Export
      │                 │                    │                   │               │
      ▼                 ▼                    ▼                   ▼               ▼
  Module docs      Validation       Transparency          Live plots      PDF/CSV/JSON
  Quick presets    Help tooltips    engine output         Console log     Jupyter export
```

**Key Components (Implemented December 2024)**:
```python
# Computation Runner
class ComputationRunner:
    def submit(params: ComputationParameters) -> str
    def get_result(job_id: str) -> ComputationResult
    def add_progress_callback(callback) -> None

# Job Queue
class JobQueueManager:
    def enqueue(params, priority) -> str
    def start_processing() -> None
    def get_queue_status() -> Dict

# Result Exporter
class ResultExporter:
    def export_json(result, path) -> bool
    def export_csv(results, path) -> bool
    def export_html(result, path) -> bool
    def export_latex(result, path) -> bool
```

### Phase 5: Visualization (Weeks 17-20) ✅ COMPLETE

**Deliverables**:
- [x] Matplotlib/PyQtGraph integration
- [x] Interactive plot widgets
- [x] RGFlowPlot for RG trajectories
- [x] SpectralDimensionPlot for d_spec flow
- [x] FixedPointPlot for basin of attraction
- [x] Export capabilities

**Key Components (Implemented December 2024)**:
```python
class RGFlowPlot:
    def add_trajectory(trajectory: RGTrajectory) -> None
    def mark_fixed_point() -> None
    def get_figure() -> Figure

class SpectralDimensionPlot:
    def plot_flow(k_values, d_spec_values) -> None
    def mark_limits() -> None

class FixedPointPlot:
    def plot_vector_field() -> None
    def add_trajectories() -> None
```

### Phase 6: Plugin System (Weeks 21-24) ✅ COMPLETE

**Deliverables**:
- [x] Plugin discovery and loading
- [x] Plugin API documentation
- [x] Example plugins (UniversalExponentPlugin, FixedPointVerifierPlugin)
- [x] Plugin manager UI support
- [x] Parameter validation

**Key Components (Implemented December 2024)**:
```python
class IRHPlugin(ABC):
    info: PluginInfo
    parameters: Dict[str, Dict]
    
    @abstractmethod
    def run(context: PluginContext, params: Dict) -> PluginResult

class PluginManager:
    def discover_plugins() -> Dict[str, PluginInfo]
    def load_plugin(name: str) -> bool
    def run_plugin(name: str, params: Dict) -> PluginResult
```

### Phase 7: Packaging (Weeks 25-28) 🚧 IN PROGRESS

**Deliverables**:
- [x] Debian package structure (basic)
- [ ] Post-install scripts
- [ ] Desktop integration (icons, menu entries)
- [ ] Man pages
- [ ] Repository hosting

---

## 6. Technical Specifications

### 6.1 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Debian 11, Ubuntu 20.04 | Debian 12, Ubuntu 22.04+ |
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 16 GB |
| Disk | 500 MB | 2 GB |
| Display | 1280×720 | 1920×1080 |

### 6.2 Dependencies

**Runtime Dependencies**:
```
python3 (>= 3.10)
python3-pyqt6 (>= 6.4)
python3-numpy (>= 1.24)
python3-scipy (>= 1.10)
python3-matplotlib (>= 3.7)
git
```

**Build Dependencies**:
```
python3-build
python3-setuptools
debhelper (>= 12)
dh-python
```

### 6.3 File System Layout

```
/opt/irh/
├── desktop/                    # Desktop application
│   ├── bin/
│   │   └── irh-desktop         # Main executable
│   ├── lib/
│   │   └── python/             # Python packages
│   └── share/
│       ├── icons/
│       ├── themes/
│       └── docs/
├── engine/                     # IRH computational engine
│   └── (cloned repository)
├── config/                     # User configuration
│   ├── settings.yaml
│   └── profiles/
├── data/                       # User data
│   ├── results/
│   └── exports/
└── plugins/                    # User plugins
```

---

## 7. Build and Packaging

### 7.1 Debian Package Structure

```
irh-desktop_21.0.0-1_amd64.deb
├── DEBIAN/
│   ├── control                 # Package metadata
│   ├── conffiles              # Configuration files list
│   ├── postinst               # Post-installation script
│   ├── prerm                  # Pre-removal script
│   └── postrm                 # Post-removal script
├── opt/
│   └── irh/
│       └── desktop/           # Application files
├── usr/
│   ├── bin/
│   │   └── irh-desktop -> /opt/irh/desktop/bin/irh-desktop
│   └── share/
│       ├── applications/
│       │   └── irh-desktop.desktop
│       ├── icons/
│       │   └── hicolor/...
│       ├── man/
│       │   └── man1/
│       │       └── irh-desktop.1
│       └── doc/
│           └── irh-desktop/
│               ├── copyright
│               └── changelog.Debian.gz
└── etc/
    └── irh/
        └── desktop.conf       # System-wide configuration
```

### 7.2 Control File

```
Package: irh-desktop
Version: 21.0.0-1
Section: science
Priority: optional
Architecture: amd64
Depends: python3 (>= 3.10), python3-pyqt6, python3-numpy, python3-scipy, 
         python3-matplotlib, git
Recommends: python3-sympy, python3-h5py
Suggests: irh-engine, jupyter
Maintainer: Brandon D. McCrary <brandon@irhresearch.org>
Homepage: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-
Description: Intrinsic Resonance Holography Desktop Interface
 IRH Desktop provides a feature-rich graphical interface for the
 Intrinsic Resonance Holography v21.0 computational framework.
 .
 Features include:
  * Automatic engine installation and updates
  * Transparent, verbose computation output
  * Interactive visualization of RG flows
  * Customizable configuration profiles
  * Plugin system for extensions
```

### 7.3 Post-Install Script

```bash
#!/bin/bash
# postinst - Post-installation script for irh-desktop

set -e

case "$1" in
    configure)
        # Create system directories
        mkdir -p /opt/irh/engine
        mkdir -p /var/log/irh
        
        # Set permissions
        chmod 755 /opt/irh/desktop/bin/irh-desktop
        
        # Update icon cache
        if command -v gtk-update-icon-cache &> /dev/null; then
            gtk-update-icon-cache -f /usr/share/icons/hicolor || true
        fi
        
        # Update desktop database
        if command -v update-desktop-database &> /dev/null; then
            update-desktop-database /usr/share/applications || true
        fi
        
        # Prompt for engine installation
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║  IRH Desktop v21.0 installed successfully!                  ║"
        echo "║                                                             ║"
        echo "║  To install the IRH engine, run:                           ║"
        echo "║    irh-desktop --setup                                      ║"
        echo "║                                                             ║"
        echo "║  Or launch from your application menu.                      ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        ;;
esac

exit 0
```

### 7.4 Build Commands

```bash
# Build Python wheel
python -m build

# Build Debian package
dpkg-buildpackage -us -uc -b

# Or using debhelper
debuild -us -uc

# Install locally
sudo dpkg -i ../irh-desktop_21.0.0-1_amd64.deb
sudo apt-get install -f  # Fix dependencies if needed
```

---

## 8. Testing Strategy

### 8.1 Test Categories

| Category | Tools | Coverage Target |
|----------|-------|-----------------|
| Unit Tests | pytest | 80% |
| UI Tests | pytest-qt | Key workflows |
| Integration | pytest | Engine integration |
| System | Manual + scripts | Installation/update |
| Usability | User testing | Key features |

### 8.2 Automated Testing

```python
# tests/test_engine_manager.py
import pytest
from irh_desktop.core.engine_manager import EngineManager

class TestEngineManager:
    def test_discover_no_engines(self, tmp_path):
        """Test discovery when no engines installed."""
        mgr = EngineManager(install_dir=tmp_path)
        engines = mgr.discover_engines()
        assert engines == []
    
    def test_install_from_github(self, tmp_path, mock_github):
        """Test engine installation from GitHub."""
        mgr = EngineManager(install_dir=tmp_path)
        result = mgr.install_engine(source="github:latest", path=tmp_path / "engine")
        assert result.success
        assert (tmp_path / "engine" / "IRH21.md").exists()
    
    def test_update_check(self, installed_engine, mock_github):
        """Test update checking."""
        mgr = EngineManager()
        update = mgr.check_update(installed_engine)
        assert update.available
        assert update.version > installed_engine.version
```

### 8.3 CI/CD Pipeline

```yaml
# .github/workflows/build-deb.yml
name: Build Debian Package

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-22.04
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install build dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y python3-build debhelper dh-python
          
      - name: Build package
        run: |
          dpkg-buildpackage -us -uc -b
          
      - name: Test installation
        run: |
          sudo dpkg -i ../irh-desktop_*.deb
          sudo apt-get install -f -y
          irh-desktop --version
          
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: irh-desktop-deb
          path: ../irh-desktop_*.deb
```

---

## 9. Distribution Plan

### 9.1 Distribution Channels

| Channel | Audience | Update Frequency |
|---------|----------|------------------|
| GitHub Releases | Early adopters | Each version |
| PPA (Ubuntu) | Ubuntu users | Stable releases |
| Direct download | All | Each version |

### 9.2 PPA Setup

```bash
# Add PPA (for end users)
sudo add-apt-repository ppa:irh/stable
sudo apt update
sudo apt install irh-desktop
```

### 9.3 Release Process

1. **Version Bump**: Update version in `pyproject.toml`, `debian/changelog`
2. **Changelog**: Document changes in `debian/changelog`
3. **Build**: Create .deb package
4. **Test**: Install on clean system, run test suite
5. **Sign**: GPG sign the package
6. **Upload**: Push to PPA and GitHub Releases
7. **Announce**: Update documentation, notify users

---

## 10. Timeline and Milestones

### 10.1 Development Schedule

```
2025 Q1: Foundation & Engine Integration (Phases 1-2)
├── Week 1-4:   Project setup, PyQt6 shell, basic UI
├── Week 5-8:   Engine discovery, installation, updates
└── Milestone:  Alpha release - basic functionality

2025 Q2: Core Features (Phases 3-4)
├── Week 9-12:  Transparency engine, output console
├── Week 13-16: Computation interface, job management
└── Milestone:  Beta release - feature complete

2025 Q3: Polish & Packaging (Phases 5-7)
├── Week 17-20: Visualization, interactive plots
├── Week 21-24: Plugin system, extensibility
├── Week 25-28: Debian packaging, distribution
└── Milestone:  Release Candidate

2025 Q4: Release & Maintenance
├── Week 29-32: Final testing, documentation
├── Week 33+:   Stable release, ongoing maintenance
└── Milestone:  v1.0.0 Stable Release
```

### 10.2 Key Milestones

| Milestone | Target Date | Criteria |
|-----------|-------------|----------|
| Alpha | 2025-03-31 | Basic installation, engine loading |
| Beta | 2025-06-30 | All features, initial documentation |
| RC | 2025-09-30 | Complete testing, packaging ready |
| v1.0 | 2025-11-30 | Stable, documented, distributed |

### 10.3 Resource Requirements

| Role | Effort | Notes |
|------|--------|-------|
| Developer | 1 FTE | Python, PyQt6, packaging |
| Designer | 0.25 FTE | UI/UX, icons, themes |
| Tester | 0.25 FTE | QA, user testing |
| Documenter | 0.25 FTE | User guide, API docs |

---

## Appendix A: Desktop File

```ini
[Desktop Entry]
Name=IRH Desktop
Comment=Intrinsic Resonance Holography Computational Framework
Exec=/opt/irh/desktop/bin/irh-desktop %F
Icon=irh-desktop
Terminal=false
Type=Application
Categories=Science;Physics;Education;
Keywords=physics;quantum;gravity;unified;theory;
MimeType=application/x-irh-config;application/x-irh-results;
StartupWMClass=irh-desktop
```

---

## Appendix B: Man Page

```
.TH IRH-DESKTOP 1 "December 2024" "IRH Desktop 21.0" "User Commands"
.SH NAME
irh-desktop \- Intrinsic Resonance Holography Desktop Interface
.SH SYNOPSIS
.B irh-desktop
[\fIOPTIONS\fR] [\fIFILE\fR]
.SH DESCRIPTION
.B irh-desktop
provides a graphical interface for the Intrinsic Resonance Holography
v21.0 computational framework. It enables interactive exploration of
the theory's predictions with full transparency about the underlying
computations.
.SH OPTIONS
.TP
.B \-\-setup
Launch the setup wizard for first-time installation
.TP
.B \-\-update
Check for and install updates
.TP
.B \-\-config \fIFILE\fR
Use specified configuration file
.TP
.B \-v, \-\-verbose
Enable verbose output
.TP
.B \-\-version
Display version information and exit
.TP
.B \-h, \-\-help
Display help message and exit
.SH FILES
.TP
.I /opt/irh/config/settings.yaml
User configuration file
.TP
.I /opt/irh/engine/
IRH computational engine installation
.TP
.I /var/log/irh/
Log files
.SH AUTHOR
Brandon D. McCrary <brandon@irhresearch.org>
.SH SEE ALSO
.BR python3 (1)
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **cGFT** | Complex quaternionic Group Field Theory |
| **Cosmic Fixed Point** | Unique IR attractor of RG flow |
| **Transparency Engine** | System for verbose, contextual output |
| **QNCD** | Quantum Normalized Compression Distance |
| **RG Flow** | Renormalization Group trajectory |

---

*This roadmap is a living document. Updates will be made as development progresses.*

**Document History**:
- v1.0 (December 2024): Initial roadmap
