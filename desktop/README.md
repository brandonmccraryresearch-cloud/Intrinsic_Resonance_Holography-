# IRH Desktop Application

Intrinsic Resonance Holography v21.0 Desktop Interface

## Overview

IRH Desktop provides a feature-rich graphical interface for the IRH computational framework, featuring:

- **Transparent Output**: All computations explain themselves with theoretical references
- **Auto-Update System**: Automatic download and installation of latest IRH engine
- **Customization**: Easy modification of parameters and configurations
- **Visualization**: Interactive plots for RG flow and emergent physics

## Building the .deb Package

### Quick Build (using dpkg-deb)

```bash
cd desktop
./build-deb.sh
```

This creates `dist/irh-desktop_21.0.0-1_all.deb`.

### Full Debian Build (optional, requires devscripts)

```bash
cd desktop
./build-deb.sh --full
```

### Build Requirements

- `dpkg-deb` (from dpkg package)
- Python 3.10+
- Optional: `debhelper`, `dh-python`, `debuild` for full build

## Installation

### From .deb Package (Debian/Ubuntu)

```bash
# Install from local build
cd desktop/dist
sudo dpkg -i irh-desktop_21.0.0-1_all.deb
sudo apt-get install -f  # Install dependencies

# Or download from releases
wget https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/releases/download/v21.0.0/irh-desktop_21.0.0-1_all.deb
sudo dpkg -i irh-desktop_21.0.0-1_all.deb
sudo apt-get install -f
```

### From Source

```bash
cd desktop
pip install -e .
```

## Usage

### Launch GUI
```bash
irh-desktop
```

### First-Time Setup
```bash
irh-desktop --setup
```

### Check for Updates
```bash
irh-desktop --update
```

### Command-Line Options
```bash
irh-desktop --help
irh-desktop --version
irh-desktop --verbose
irh-desktop --config /path/to/config.yaml
```

## Features

### Module Navigator
Browse and run different computation modules:
- Primitives (Quaternions, Group Manifolds, QNCD)
- cGFT Actions
- RG Flow (Beta Functions, Fixed Points)
- Emergent Spacetime
- Topology (Betti Numbers, Instantons)
- Standard Model
- Predictions

### Transparency Console
All computations display verbose output with:
- Step-by-step progress
- Equation references (e.g., "Eq. 1.13")
- Numerical values with precision
- Plain-language explanations
- Verification results

### Configuration Profiles
Save and load different computation configurations:
- Lattice settings
- RG flow parameters
- Precision options
- Output verbosity

## Architecture

```
desktop/
├── src/irh_desktop/
│   ├── main.py          # Entry point
│   ├── app.py           # Qt application
│   ├── core/
│   │   ├── engine_manager.py    # Engine lifecycle
│   │   └── config_manager.py    # Configuration
│   ├── transparency/
│   │   └── engine.py    # Transparent output
│   └── ui/
│       ├── main_window.py       # Main window
│       └── setup_wizard.py      # Setup wizard
├── debian/              # Debian packaging
└── resources/           # Icons, themes
```

## License

MIT License - See LICENSE file
