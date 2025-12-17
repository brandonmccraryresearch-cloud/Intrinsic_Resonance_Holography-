#!/bin/bash
# ==============================================================================
# Intrinsic Resonance Holography (IRH) v21.1 - Installation Script
# 
# THEORETICAL FOUNDATION: Intrinsic_Resonance_Holography-v21.1.md §1.6
# Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-
#
# This script installs IRH and all dependencies on Linux/macOS systems.
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-.git"
INSTALL_DIR="${IRH_INSTALL_DIR:-$HOME/IRH}"
VENV_NAME="irh_venv"
PYTHON_MIN_VERSION="3.10"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║     Intrinsic Resonance Holography (IRH) v21.1 - Installation Script       ║"
    echo "║                                                                            ║"
    echo "║  Unified Theory of Emergent Reality from Quantum-Informational Principles  ║"
    echo "║  https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography- ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[i] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[✗] $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

version_gte() {
    # Check if version $1 >= version $2
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

# ==============================================================================
# Prerequisite Checks
# ==============================================================================

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python ${PYTHON_MIN_VERSION}+ first."
        echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        echo "  macOS: brew install python@3.11"
        echo "  Fedora: sudo dnf install python3 python3-pip"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if ! version_gte "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
        print_error "Python ${PYTHON_MIN_VERSION}+ is required. Found: ${PYTHON_VERSION}"
        exit 1
    fi
    print_step "Python ${PYTHON_VERSION} found"
    
    # Check pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip is not installed. Installing..."
        $PYTHON_CMD -m ensurepip --default-pip || {
            print_error "Could not install pip. Please install manually."
            exit 1
        }
    fi
    print_step "pip is available"
    
    # Check venv
    if ! $PYTHON_CMD -c "import venv" &> /dev/null; then
        print_warning "venv module not found. Attempting to install..."
        if check_command apt; then
            sudo apt install -y python3-venv
        elif check_command dnf; then
            sudo dnf install -y python3-venv
        else
            print_error "Could not install venv. Please install python3-venv manually."
            exit 1
        fi
    fi
    print_step "venv module is available"
    
    # Check git
    if ! check_command git; then
        print_error "Git is not installed. Please install Git first."
        echo "  Ubuntu/Debian: sudo apt install git"
        echo "  macOS: brew install git"
        echo "  Fedora: sudo dnf install git"
        exit 1
    fi
    print_step "Git is available"
}

# ==============================================================================
# Installation Functions
# ==============================================================================

create_install_directory() {
    print_info "Creating installation directory at: ${INSTALL_DIR}"
    
    if [ -d "$INSTALL_DIR" ]; then
        print_warning "Directory already exists. Updating existing installation..."
    else
        mkdir -p "$INSTALL_DIR"
    fi
    
    print_step "Installation directory ready"
}

clone_repository() {
    print_info "Cloning IRH repository..."
    
    if [ -d "$INSTALL_DIR/.git" ]; then
        print_info "Repository exists. Pulling latest changes..."
        cd "$INSTALL_DIR"
        git pull origin main || git pull origin master
    else
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    
    print_step "Repository cloned/updated"
}

create_virtual_environment() {
    print_info "Creating Python virtual environment..."
    
    cd "$INSTALL_DIR"
    
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment exists. Using existing one..."
    else
        $PYTHON_CMD -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_step "Virtual environment created and activated"
}

install_dependencies() {
    print_info "Installing dependencies..."
    
    cd "$INSTALL_DIR"
    source "$VENV_NAME/bin/activate"
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found. Installing core dependencies..."
        pip install numpy scipy sympy matplotlib jupyter pytest
    fi
    
    print_step "Dependencies installed"
}

verify_installation() {
    print_info "Verifying installation..."
    
    cd "$INSTALL_DIR"
    source "$VENV_NAME/bin/activate"
    
    # Test imports
    $PYTHON_CMD -c "
from src.primitives.quaternions import Quaternion
from src.rg_flow.fixed_points import find_fixed_point
print('✓ Core modules imported successfully')

# Quick verification
fp = find_fixed_point()
print(f'✓ Cosmic Fixed Point: λ̃* = {fp.lambda_star:.3f}')
print('✓ Installation verified!')
" || {
        print_error "Installation verification failed. Check the error messages above."
        exit 1
    }
    
    print_step "Installation verified successfully"
}

create_launcher_script() {
    print_info "Creating launcher script..."
    
    cd "$INSTALL_DIR"
    
    # Create launcher script
    cat > irh << 'EOF'
#!/bin/bash
# IRH Launcher Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/irh_venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

case "$1" in
    "test")
        pytest tests/ -v "${@:2}"
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
        ;;
    *)
        echo "IRH v21.1 - Intrinsic Resonance Holography"
        echo ""
        echo "Usage: irh <command>"
        echo ""
        echo "Commands:"
        echo "  test       Run the test suite"
        echo "  notebook   Launch Jupyter Lab with notebooks"
        echo "  verify     Verify theoretical annotations"
        echo "  compute    Compute all observables"
        echo "  shell      Open Python shell with IRH loaded"
        echo ""
        echo "Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-"
        ;;
esac
EOF
    
    chmod +x irh
    
    print_step "Launcher script created"
}

setup_shell_integration() {
    print_info "Setting up shell integration..."
    
    # Detect shell
    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi
    
    if [ -n "$SHELL_RC" ]; then
        # Add to PATH if not already there
        if ! grep -q "IRH_HOME" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Intrinsic Resonance Holography (IRH)" >> "$SHELL_RC"
            echo "export IRH_HOME=\"$INSTALL_DIR\"" >> "$SHELL_RC"
            echo "export PATH=\"\$IRH_HOME:\$PATH\"" >> "$SHELL_RC"
            print_step "Added IRH to PATH in $SHELL_RC"
            print_warning "Please restart your shell or run: source $SHELL_RC"
        else
            print_info "IRH already in shell configuration"
        fi
    else
        print_warning "Could not detect shell configuration file. Add manually:"
        echo "  export IRH_HOME=\"$INSTALL_DIR\""
        echo "  export PATH=\"\$IRH_HOME:\$PATH\""
    fi
}

# ==============================================================================
# Main Installation Flow
# ==============================================================================

main() {
    print_header
    
    echo ""
    print_info "Installation directory: ${INSTALL_DIR}"
    print_info "You can change this by setting IRH_INSTALL_DIR environment variable"
    echo ""
    
    # Ask for confirmation
    read -p "Proceed with installation? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
        print_info "Installation cancelled."
        exit 0
    fi
    
    echo ""
    
    # Run installation steps
    check_prerequisites
    create_install_directory
    clone_repository
    create_virtual_environment
    install_dependencies
    verify_installation
    create_launcher_script
    setup_shell_integration
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Installation Complete!                                  ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Quick Start:"
    echo "  1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    echo "  2. Run: irh test           # Run the test suite"
    echo "  3. Run: irh notebook       # Launch Jupyter notebooks"
    echo "  4. Run: irh shell          # Python shell with IRH"
    echo ""
    echo "Documentation:"
    echo "  - README: ${INSTALL_DIR}/README.md"
    echo "  - Theory: ${INSTALL_DIR}/Intrinsic_Resonance_Holography-v21.1.md"
    echo "  - Notebooks: ${INSTALL_DIR}/notebooks/"
    echo ""
    echo "Repository: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-"
    echo ""
}

# Run main function
main "$@"
