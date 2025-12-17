#!/usr/bin/env python3
"""
IRH Desktop Application - Main Entry Point

This module provides the entry point for the IRH Desktop application.
It handles command-line arguments, initialization, and launching the GUI.

Usage:
    irh-desktop                    # Launch GUI
    irh-desktop --setup            # Run setup wizard
    irh-desktop --update           # Check for updates
    irh-desktop --version          # Show version
    irh-desktop --verbose          # Enable verbose mode

Theoretical Foundation:
    IRH21.md - Intrinsic Resonance Holography v21.0
    
Author: Brandon D. McCrary
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='irh-desktop',
        description='Intrinsic Resonance Holography Desktop Application',
        epilog='For more information, visit: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-'
    )
    
    parser.add_argument(
        '--version', '-V',
        action='store_true',
        help='Show version information and exit'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Launch the setup wizard for first-time installation'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Check for and install updates'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        metavar='FILE',
        help='Use specified configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with extra logging'
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in headless mode (no GUI)'
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        type=str,
        help='Open specified configuration or results file'
    )
    
    return parser.parse_args()


def show_version() -> None:
    """Display version information."""
    from irh_desktop import __version__
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    IRH Desktop Application v{__version__}                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Intrinsic Resonance Holography v21.0 Computational Framework            ║
║                                                                          ║
║  A unified theory deriving fundamental constants from first principles   ║
║  using quaternionic Group Field Theory on G_inf = SU(2) × U(1)_φ         ║
║                                                                          ║
║  Key Predictions:                                                        ║
║  • α⁻¹ = 137.035999... (fine-structure constant)                        ║
║  • w₀ = -0.91234567 (dark energy equation of state)                     ║
║  • β₁ = 12 → SU(3)×SU(2)×U(1) gauge group                               ║
║  • n_inst = 3 → Three fermion generations                               ║
║                                                                          ║
║  Author: Brandon D. McCrary                                              ║
║  License: MIT                                                            ║
║  Repository: github.com/brandonmccraryresearch-cloud/                    ║
║              Intrinsic_Resonance_Holography-                              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def run_setup_wizard() -> int:
    """
    Run the first-time setup wizard.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    logger.info("Starting IRH Desktop Setup Wizard...")
    
    from irh_desktop.ui.setup_wizard import SetupWizard
    from irh_desktop.app import create_app
    
    app = create_app(sys.argv)
    wizard = SetupWizard()
    wizard.show()
    return app.exec()


def check_updates() -> int:
    """
    Check for and optionally install updates.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    logger.info("Checking for IRH Engine updates...")
    
    from irh_desktop.core.engine_manager import EngineManager
    
    manager = EngineManager()
    update_info = manager.check_for_updates()
    
    if update_info.available:
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         Update Available                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Current Version: {update_info.current_version:<20}                              ║
║  Available Version: {update_info.latest_version:<18}                              ║
║                                                                          ║
║  Changes:                                                                ║
""")
        for change in update_info.changelog[:5]:
            print(f"║  • {change:<66} ║")
        print("""║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
        
        response = input("Install update? [y/N]: ").strip().lower()
        if response == 'y':
            success = manager.install_update(update_info)
            if success:
                logger.info("Update installed successfully!")
                return 0
            else:
                logger.error("Update installation failed.")
                return 1
    else:
        print("✓ IRH Engine is up to date.")
    
    return 0


def launch_gui(args: argparse.Namespace) -> int:
    """
    Launch the main GUI application.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns
    -------
    int
        Exit code from application
    """
    logger.info("Launching IRH Desktop Application...")
    
    from irh_desktop.app import create_app
    from irh_desktop.ui.main_window import MainWindow
    from irh_desktop.core.config_manager import ConfigManager
    
    # Initialize application
    app = create_app(sys.argv)
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        config_manager.load(args.config)
    
    # Create main window
    main_window = MainWindow(
        config_manager=config_manager,
        verbose=args.verbose,
        debug=args.debug
    )
    
    # Open file if specified
    if args.file:
        main_window.open_file(args.file)
    
    # Show window and run event loop
    main_window.show()
    return app.exec()


def main() -> int:
    """
    Main entry point for IRH Desktop.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    args = parse_arguments()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Handle version request
    if args.version:
        show_version()
        return 0
    
    # Handle setup wizard
    if args.setup:
        return run_setup_wizard()
    
    # Handle update check
    if args.update:
        return check_updates()
    
    # Handle headless mode
    if args.no_gui:
        logger.info("Running in headless mode")
        # Could run computations without GUI here
        return 0
    
    # Launch main GUI
    return launch_gui(args)


if __name__ == "__main__":
    sys.exit(main())
