# IRH Installation Scripts

This directory contains installation scripts for Intrinsic Resonance Holography (IRH) v21.1.

## Quick Start

### Linux/macOS

```bash
# Download and run the installer
curl -sSL https://raw.githubusercontent.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-/main/installers/install.sh | bash

# Or clone and run locally
git clone https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-.git
cd Intrinsic_Resonace_Holography-/installers
chmod +x install.sh
./install.sh
```

### Windows

```cmd
REM Download install.bat and run it
REM Or use the Python installer:
python install.py
```

### Cross-Platform (Python)

```bash
python install.py [--dir <install_dir>] [--dev]
```

## Available Scripts

| Script | Platform | Description |
|--------|----------|-------------|
| `install.sh` | Linux/macOS | Shell script installer |
| `install.bat` | Windows | Batch file installer |
| `install.py` | All | Cross-platform Python installer |
| `irh_setup.iss` | Windows | Inno Setup script for .exe creation |

## Building Windows Installer (.exe)

To create a Windows installer executable:

1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `irh_setup.iss` in Inno Setup Compiler
3. Click Build > Compile (or press Ctrl+F9)
4. The installer will be created in the `Output` directory

## Installation Options

### install.sh Options

```bash
# Set custom install directory
export IRH_INSTALL_DIR=/opt/irh
./install.sh
```

### install.py Options

```bash
python install.py --help

Options:
  --dir, -d <path>  Installation directory (default: ~/IRH)
  --no-venv         Skip virtual environment creation
  --dev             Install development dependencies
```

## Post-Installation

After installation, use the `irh` command:

```bash
irh test       # Run tests
irh notebook   # Launch Jupyter notebooks
irh verify     # Verify theoretical annotations
irh compute    # Compute all observables
irh shell      # Python shell with IRH
```

## Requirements

- **Python**: 3.10 or later
- **Git**: For cloning/updating the repository
- **OS**: Linux, macOS, or Windows

## Support

- **Repository**: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-
- **Issues**: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-/issues

## Citation

```bibtex
@software{IRH_v21_computational_2025,
  title={Intrinsic Resonance Holography v21.1: Computational Framework},
  author={McCrary, Brandon D.},
  year={2025},
  url={https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonace_Holography-}
}
```
