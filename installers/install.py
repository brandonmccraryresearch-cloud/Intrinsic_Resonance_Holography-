#!/usr/bin/env python3
"""
Intrinsic Resonance Holography (IRH) v21.1 - Cross-Platform Python Installer

THEORETICAL FOUNDATION: IRH v21.1 Manuscript (Part 1 §1-4, Part 2 §5-8)
Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-

This script provides a cross-platform installation method for IRH.
Works on Windows, Linux, and macOS.

Components:
    - Core computational framework (970+ tests)
    - Web interface (FastAPI + React)
    - ML surrogate models
    - Experimental data pipeline with online updates
    - Desktop application (PyQt6)

Usage:
    python install.py [--dir <install_dir>] [--no-venv] [--dev] [--webapp] [--full]
    
Options:
    --dir <path>    Installation directory (default: ~/IRH)
    --no-venv       Skip virtual environment creation
    --dev           Install development dependencies
    --webapp        Install web application dependencies
    --full          Full installation with all components
    --help          Show this help message
"""

import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path
import venv
from typing import Optional

# ==============================================================================
# Configuration
# ==============================================================================

REPO_URL = "https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-.git"
MIN_PYTHON_VERSION = (3, 10)
GITHUB_CITATION = "https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-"

# ==============================================================================
# Color Output
# ==============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-terminal output)."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ''
        cls.MAGENTA = cls.CYAN = cls.WHITE = cls.BOLD = cls.END = ''

# Disable colors on Windows without ANSI support
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        Colors.disable()


def print_header():
    """Print installation header."""
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║     {Colors.BOLD}Intrinsic Resonance Holography (IRH) v21.1 - Python Installer{Colors.END}{Colors.CYAN}          ║
║                                                                              ║
║  Unified Theory of Emergent Reality from Quantum-Informational Principles    ║
║  {GITHUB_CITATION}  ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")


def print_step(msg: str):
    """Print a completed step."""
    print(f"{Colors.GREEN}[✓] {msg}{Colors.END}")


def print_info(msg: str):
    """Print an info message."""
    print(f"{Colors.BLUE}[i] {msg}{Colors.END}")


def print_warning(msg: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}[!] {msg}{Colors.END}")


def print_error(msg: str):
    """Print an error message."""
    print(f"{Colors.RED}[✗] {msg}{Colors.END}")


# ==============================================================================
# Installation Functions
# ==============================================================================

def check_python_version() -> bool:
    """Check if Python version meets minimum requirements."""
    current = sys.version_info[:2]
    if current < MIN_PYTHON_VERSION:
        print_error(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required. "
                   f"Found: {current[0]}.{current[1]}")
        return False
    print_step(f"Python {current[0]}.{current[1]} found")
    return True


def check_git() -> bool:
    """Check if git is installed."""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        print_step("Git is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Git is not installed. Please install Git first.")
        print(f"  - Windows: https://git-scm.com/download/win")
        print(f"  - macOS: brew install git")
        print(f"  - Linux: sudo apt install git (or dnf/pacman)")
        return False


def clone_or_update_repo(install_dir: Path) -> bool:
    """Clone the repository or update if it exists."""
    print_info("Setting up repository...")
    
    git_dir = install_dir / '.git'
    
    try:
        if git_dir.exists():
            print_info("Repository exists. Pulling latest changes...")
            subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                cwd=install_dir,
                capture_output=True
            )
        else:
            print_info(f"Cloning repository to {install_dir}...")
            subprocess.run(
                ['git', 'clone', REPO_URL, str(install_dir)],
                check=True
            )
        print_step("Repository ready")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Git operation failed: {e}")
        return False


def create_virtual_environment(install_dir: Path, venv_name: str = "irh_venv") -> Optional[Path]:
    """Create a Python virtual environment."""
    print_info("Creating virtual environment...")
    
    venv_path = install_dir / venv_name
    
    try:
        if venv_path.exists():
            print_warning("Virtual environment exists. Using existing one...")
        else:
            venv.create(venv_path, with_pip=True)
        
        print_step("Virtual environment created")
        return venv_path
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return None


def get_pip_executable(venv_path: Path) -> Path:
    """Get the pip executable path for the virtual environment."""
    if sys.platform == 'win32':
        return venv_path / 'Scripts' / 'pip.exe'
    return venv_path / 'bin' / 'pip'


def get_python_executable(venv_path: Path) -> Path:
    """Get the Python executable path for the virtual environment."""
    if sys.platform == 'win32':
        return venv_path / 'Scripts' / 'python.exe'
    return venv_path / 'bin' / 'python'


def install_dependencies(install_dir: Path, venv_path: Path, dev: bool = False, webapp: bool = False) -> bool:
    """Install Python dependencies."""
    print_info("Installing dependencies...")
    
    pip = get_pip_executable(venv_path)
    
    try:
        # Upgrade pip
        subprocess.run([str(pip), 'install', '--upgrade', 'pip'], check=True, capture_output=True)
        
        # Install requirements
        requirements_file = install_dir / 'requirements.txt'
        if requirements_file.exists():
            subprocess.run([str(pip), 'install', '-r', str(requirements_file)], check=True)
        else:
            print_warning("requirements.txt not found. Installing core dependencies...")
            core_deps = ['numpy', 'scipy', 'sympy', 'matplotlib', 'jupyter', 'pytest']
            subprocess.run([str(pip), 'install'] + core_deps, check=True)
        
        # Install dev dependencies if requested
        if dev:
            print_info("Installing development dependencies...")
            dev_deps = ['black', 'isort', 'flake8', 'mypy', 'pytest-cov', 'sphinx', 'hypothesis']
            subprocess.run([str(pip), 'install'] + dev_deps, check=True)
        
        # Install webapp dependencies if requested
        if webapp:
            print_info("Installing web application dependencies...")
            webapp_deps = ['fastapi', 'uvicorn', 'pydantic', 'python-multipart']
            subprocess.run([str(pip), 'install'] + webapp_deps, check=True)
            
            # Check for Node.js for frontend
            try:
                subprocess.run(['node', '--version'], capture_output=True, check=True)
                frontend_dir = install_dir / 'webapp' / 'frontend'
                if frontend_dir.exists():
                    print_info("Installing frontend dependencies with npm...")
                    subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print_warning("Node.js not found. Frontend dependencies not installed.")
                print_warning("To install frontend: npm install (in webapp/frontend)")
        
        print_step("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def verify_installation(install_dir: Path, venv_path: Path) -> bool:
    """Verify the installation by running a quick test."""
    print_info("Verifying installation...")
    
    python = get_python_executable(venv_path)
    
    test_code = '''
import sys
sys.path.insert(0, '.')
from src.primitives.quaternions import Quaternion
from src.rg_flow.fixed_points import find_fixed_point
fp = find_fixed_point()
print(f"Cosmic Fixed Point: λ̃* = {fp.lambda_star:.3f}")
print("Installation verified!")
'''
    
    try:
        result = subprocess.run(
            [str(python), '-c', test_code],
            cwd=install_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_step("Installation verified successfully")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
            return True
        else:
            print_error("Verification failed")
            print(result.stderr)
            return False
    except Exception as e:
        print_error(f"Verification error: {e}")
        return False


def create_launcher_scripts(install_dir: Path, venv_path: Path, webapp: bool = False):
    """Create launcher scripts for easy access."""
    print_info("Creating launcher scripts...")
    
    if sys.platform == 'win32':
        # Windows batch script
        launcher = install_dir / 'irh.bat'
        webapp_section = """
if "%1"=="webapp" (
    echo Starting IRH Web API...
    cd /d "%IRH_HOME%\\webapp\\backend"
    uvicorn app:app --reload --port 8000
    goto :eof
)
if "%1"=="frontend" (
    echo Starting IRH Frontend...
    cd /d "%IRH_HOME%\\webapp\\frontend"
    npm run dev
    goto :eof
)
if "%1"=="update-data" (
    echo Updating experimental data...
    python -c "from src.experimental import update_codata_online, update_pdg_online; print(update_codata_online()); print(update_pdg_online())"
    goto :eof
)""" if webapp else ""
        
        content = f'''@echo off
setlocal
set "IRH_HOME={install_dir}"
call "%IRH_HOME%\\{venv_path.name}\\Scripts\\activate.bat"
set "PYTHONPATH=%IRH_HOME%;%PYTHONPATH%"

if "%1"=="test" (
    pytest tests/ -v %2 %3 %4 %5
    goto :eof
)
if "%1"=="notebook" (
    jupyter lab notebooks/
    goto :eof
)
if "%1"=="verify" (
    python scripts/verify_theoretical_annotations.py
    goto :eof
)
if "%1"=="compute" (
    python scripts/compute_all_observables.py
    goto :eof
)
if "%1"=="shell" (
    python
    goto :eof
){webapp_section}

echo IRH v21.1 - Intrinsic Resonance Holography
echo.
echo Usage: irh ^<command^>
echo.
echo Commands:
echo   test         Run the test suite (970+ tests)
echo   notebook     Launch Jupyter Lab with notebooks
echo   verify       Verify theoretical annotations
echo   compute      Compute all observables
echo   shell        Open Python shell with IRH loaded
{"echo   webapp       Start web API server (FastAPI)" if webapp else ""}
{"echo   frontend     Start frontend dev server (React)" if webapp else ""}
echo   update-data  Update PDG/CODATA experimental data
echo.
echo Notebooks:
echo   notebooks/00_quickstart.ipynb             - Quick introduction
echo   notebooks/02_rg_flow_interactive.ipynb    - RG flow explorer
echo   notebooks/05_full_stack_execution.ipynb   - Complete demo
echo.
echo Repository: {GITHUB_CITATION}
'''
    else:
        # Unix shell script
        launcher = install_dir / 'irh'
        webapp_section = '''
    "webapp")
        echo "Starting IRH Web Application..."
        cd "$IRH_HOME/webapp/backend"
        uvicorn app:app --reload --port 8000
        ;;
    "frontend")
        echo "Starting IRH Frontend..."
        cd "$IRH_HOME/webapp/frontend"
        npm run dev
        ;;
    "update-data")
        echo "Updating experimental data from PDG/CODATA..."
        python -c "from src.experimental import update_codata_online, update_pdg_online; print(update_codata_online()); print(update_pdg_online())"
        ;;''' if webapp else ""
        
        content = f'''#!/bin/bash
IRH_HOME="{install_dir}"
source "$IRH_HOME/{venv_path.name}/bin/activate"
export PYTHONPATH="$IRH_HOME:$PYTHONPATH"

case "$1" in
    "test")
        pytest tests/ -v "${{@:2}}"
        ;;
    "notebook")
        jupyter lab notebooks/
        ;;
    "verify")
        python scripts/verify_theoretical_annotations.py
        ;;
    "compute")
        python scripts/compute_all_observables.py
        ;;
    "shell")
        python
        ;;{webapp_section}
    *)
        echo "IRH v21.1 - Intrinsic Resonance Holography"
        echo ""
        echo "Usage: irh <command>"
        echo ""
        echo "Commands:"
        echo "  test         Run the test suite (970+ tests)"
        echo "  notebook     Launch Jupyter Lab with notebooks"
        echo "  verify       Verify theoretical annotations"
        echo "  compute      Compute all observables"
        echo "  shell        Open Python shell with IRH loaded"
{"        echo \"  webapp       Start web API server (FastAPI)\"" if webapp else ""}
{"        echo \"  frontend     Start frontend dev server (React)\"" if webapp else ""}
        echo "  update-data  Update PDG/CODATA experimental data"
        echo ""
        echo "Notebooks:"
        echo "  notebooks/00_quickstart.ipynb             - Quick introduction"
        echo "  notebooks/02_rg_flow_interactive.ipynb    - RG flow explorer"
        echo "  notebooks/05_full_stack_execution.ipynb   - Complete demo"
        echo ""
        echo "Repository: {GITHUB_CITATION}"
        ;;
esac
'''
    
    with open(launcher, 'w') as f:
        f.write(content)
    
    if sys.platform != 'win32':
        os.chmod(launcher, 0o755)
    
    print_step("Launcher scripts created")


def print_completion_message(install_dir: Path, webapp: bool = False):
    """Print completion message with next steps."""
    webapp_msg = """
  5. Run: ./irh webapp         # Start web API (port 8000)
  6. Run: ./irh frontend       # Start React frontend (port 3000)""" if webapp else ""
    
    webapp_doc = """
  - Web API docs: http://localhost:8000/docs (after starting webapp)""" if webapp else ""
    
    print(f"""
{Colors.GREEN}╔══════════════════════════════════════════════════════════════════════════════╗
║                          Installation Complete!                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.BOLD}Quick Start:{Colors.END}
  1. Navigate to: cd {install_dir}
  2. Run: ./irh test           # Run the test suite (970+ tests, Linux/macOS)
     or:  irh test             # Run the test suite (Windows)
  3. Run: ./irh notebook       # Launch Jupyter notebooks
  4. Run: ./irh shell          # Python shell with IRH{webapp_msg}

{Colors.BOLD}Recommended Notebooks:{Colors.END}
  - notebooks/00_quickstart.ipynb             # Quick introduction
  - notebooks/02_rg_flow_interactive.ipynb    # RG flow explorer
  - notebooks/05_full_stack_execution.ipynb   # Complete IRH demo

{Colors.BOLD}Documentation:{Colors.END}
  - README: {install_dir}/README.md
  - Theory: {install_dir}/Intrinsic_Resonance_Holography-v21.1-Part1.md
  - Theory: {install_dir}/Intrinsic_Resonance_Holography-v21.1-Part2.md
  - Notebooks: {install_dir}/notebooks/{webapp_doc}

{Colors.BOLD}Citation:{Colors.END}
  {GITHUB_CITATION}

{Colors.CYAN}Thank you for installing IRH - Intrinsic Resonance Holography!{Colors.END}
""")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description='Intrinsic Resonance Holography (IRH) v21.1 Installer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Components:
  - Core computational framework (970+ tests)
  - Web interface (FastAPI + React)
  - ML surrogate models
  - Experimental data pipeline with online updates
  - Desktop application (PyQt6)

Examples:
  python install.py                    # Install to ~/IRH
  python install.py --dir /opt/irh     # Install to /opt/irh
  python install.py --dev              # Include dev dependencies
  python install.py --webapp           # Include web app dependencies
  python install.py --full             # Full installation

Repository: {GITHUB_CITATION}
'''
    )
    parser.add_argument(
        '--dir', '-d',
        type=Path,
        default=Path.home() / 'IRH',
        help='Installation directory (default: ~/IRH)'
    )
    parser.add_argument(
        '--no-venv',
        action='store_true',
        help='Skip virtual environment creation'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Install development dependencies'
    )
    parser.add_argument(
        '--webapp',
        action='store_true',
        help='Install web application dependencies'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full installation with all components'
    )
    
    args = parser.parse_args()
    install_dir = args.dir.resolve()
    
    # Handle --full option
    if args.full:
        args.dev = True
        args.webapp = True
    
    print_header()
    print_info(f"Installation directory: {install_dir}")
    print_info(f"Development mode: {args.dev}")
    print_info(f"Web application: {args.webapp}")
    print()
    
    # Confirmation
    response = input("Proceed with installation? [Y/n] ").strip().lower()
    if response and response not in ('y', 'yes'):
        print_info("Installation cancelled.")
        return
    
    print()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_git():
        sys.exit(1)
    
    # Create install directory
    install_dir.mkdir(parents=True, exist_ok=True)
    print_step(f"Installation directory created: {install_dir}")
    
    # Clone repository
    if not clone_or_update_repo(install_dir):
        sys.exit(1)
    
    # Create virtual environment
    venv_path = None
    if not args.no_venv:
        venv_path = create_virtual_environment(install_dir)
        if venv_path is None:
            sys.exit(1)
        
        # Install dependencies
        if not install_dependencies(install_dir, venv_path, args.dev, args.webapp):
            sys.exit(1)
        
        # Verify installation
        if not verify_installation(install_dir, venv_path):
            print_warning("Verification failed, but installation may still work.")
        
        # Create launcher scripts
        create_launcher_scripts(install_dir, venv_path, args.webapp)
    else:
        print_warning("Skipping virtual environment. You'll need to install dependencies manually.")
    
    # Print completion message
    print_completion_message(install_dir, args.webapp)


if __name__ == '__main__':
    main()
