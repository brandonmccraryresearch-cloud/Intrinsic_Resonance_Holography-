"""
IRH Desktop - Setup Wizard

First-time setup wizard for installing and configuring the IRH engine.

Author: Brandon D. McCrary
"""

import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import PyQt6
try:
    from PyQt6.QtWidgets import (
        QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
        QLabel, QRadioButton, QButtonGroup, QLineEdit,
        QPushButton, QFileDialog, QCheckBox, QProgressBar,
        QTextEdit, QGroupBox, QApplication,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    QWizard = object


if HAS_PYQT6:
    
    class InstallWorker(QThread):
        """Background worker for engine installation."""
        
        progress = pyqtSignal(int, str)  # percent, message
        finished = pyqtSignal(bool, str)  # success, message
        
        def __init__(self, source: str, path: Path):
            super().__init__()
            self.source = source
            self.path = path
        
        def run(self):
            """Run the installation."""
            from irh_desktop.core.engine_manager import EngineManager
            
            manager = EngineManager()
            
            def progress_callback(percent, message):
                self.progress.emit(percent, message)
            
            success = manager.install_engine(
                source=self.source,
                path=self.path,
                progress_callback=progress_callback
            )
            
            if success:
                self.finished.emit(True, "Installation completed successfully!")
            else:
                self.finished.emit(False, "Installation failed. See log for details.")
    
    
    class WelcomePage(QWizardPage):
        """Welcome page of the setup wizard."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setTitle("Welcome to IRH Desktop")
            self.setSubTitle("This wizard will help you set up the IRH computational engine.")
            
            layout = QVBoxLayout(self)
            
            welcome_text = QLabel("""
            <p><b>Intrinsic Resonance Holography v21.0</b></p>
            
            <p>A comprehensive computational framework for deriving fundamental physics
            from first principles using quaternionic Group Field Theory.</p>
            
            <p><b>Key Features:</b></p>
            <ul>
            <li>Derives fundamental constants (α, w₀) from first principles</li>
            <li>Transparent, verbose output explaining all computations</li>
            <li>Complete Standard Model derivation from topology</li>
            <li>Falsifiable predictions for experimental tests</li>
            </ul>
            
            <p>Click <b>Next</b> to begin setup.</p>
            """)
            welcome_text.setWordWrap(True)
            layout.addWidget(welcome_text)
    
    
    class InstallSourcePage(QWizardPage):
        """Page for selecting installation source."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setTitle("Select Installation Source")
            self.setSubTitle("Choose how to install the IRH computational engine.")
            
            layout = QVBoxLayout(self)
            
            # Source options
            self.source_group = QButtonGroup(self)
            
            self.github_radio = QRadioButton("Download from GitHub (recommended)")
            self.github_radio.setChecked(True)
            self.source_group.addButton(self.github_radio)
            layout.addWidget(self.github_radio)
            
            github_desc = QLabel("    Download the latest version from the official repository.")
            github_desc.setStyleSheet("color: gray;")
            layout.addWidget(github_desc)
            
            self.bundled_radio = QRadioButton("Use bundled engine")
            self.source_group.addButton(self.bundled_radio)
            layout.addWidget(self.bundled_radio)
            
            bundled_desc = QLabel("    Use the engine version included with this installer.")
            bundled_desc.setStyleSheet("color: gray;")
            layout.addWidget(bundled_desc)
            
            self.local_radio = QRadioButton("Use existing installation")
            self.source_group.addButton(self.local_radio)
            layout.addWidget(self.local_radio)
            
            # Local path selection
            local_layout = QHBoxLayout()
            local_layout.addSpacing(20)
            self.local_path = QLineEdit()
            self.local_path.setPlaceholderText("Path to existing IRH installation...")
            self.local_path.setEnabled(False)
            local_layout.addWidget(self.local_path)
            
            self.browse_btn = QPushButton("Browse...")
            self.browse_btn.setEnabled(False)
            self.browse_btn.clicked.connect(self.browse_local)
            local_layout.addWidget(self.browse_btn)
            
            layout.addLayout(local_layout)
            
            # Connect radio buttons
            self.local_radio.toggled.connect(self.local_path.setEnabled)
            self.local_radio.toggled.connect(self.browse_btn.setEnabled)
            
            layout.addStretch()
            
            # Register fields
            self.registerField("github_source", self.github_radio)
            self.registerField("bundled_source", self.bundled_radio)
            self.registerField("local_source", self.local_radio)
            self.registerField("local_path", self.local_path)
        
        def browse_local(self):
            """Browse for local installation."""
            path = QFileDialog.getExistingDirectory(
                self, "Select IRH Installation Directory"
            )
            if path:
                self.local_path.setText(path)
    
    
    class InstallLocationPage(QWizardPage):
        """Page for selecting installation location."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setTitle("Installation Location")
            self.setSubTitle("Choose where to install the IRH engine.")
            
            layout = QVBoxLayout(self)
            
            # Installation directory
            dir_group = QGroupBox("Installation Directory")
            dir_layout = QHBoxLayout(dir_group)
            
            self.install_path = QLineEdit()
            default_path = Path.home() / ".local/share/irh/engine"
            self.install_path.setText(str(default_path))
            dir_layout.addWidget(self.install_path)
            
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(self.browse_install_dir)
            dir_layout.addWidget(browse_btn)
            
            layout.addWidget(dir_group)
            
            # Options
            options_group = QGroupBox("Options")
            options_layout = QVBoxLayout(options_group)
            
            self.desktop_shortcut = QCheckBox("Create desktop shortcut")
            self.desktop_shortcut.setChecked(True)
            options_layout.addWidget(self.desktop_shortcut)
            
            self.add_to_path = QCheckBox("Add to system PATH")
            self.add_to_path.setChecked(True)
            options_layout.addWidget(self.add_to_path)
            
            self.jupyter_integration = QCheckBox("Install Jupyter integration")
            self.jupyter_integration.setChecked(True)
            options_layout.addWidget(self.jupyter_integration)
            
            self.auto_updates = QCheckBox("Enable automatic updates")
            self.auto_updates.setChecked(True)
            options_layout.addWidget(self.auto_updates)
            
            layout.addWidget(options_group)
            
            layout.addStretch()
            
            # Register fields
            self.registerField("install_path", self.install_path)
            self.registerField("desktop_shortcut", self.desktop_shortcut)
            self.registerField("add_to_path", self.add_to_path)
            self.registerField("jupyter_integration", self.jupyter_integration)
            self.registerField("auto_updates", self.auto_updates)
        
        def browse_install_dir(self):
            """Browse for installation directory."""
            path = QFileDialog.getExistingDirectory(
                self, "Select Installation Directory"
            )
            if path:
                self.install_path.setText(path)
    
    
    class InstallProgressPage(QWizardPage):
        """Page showing installation progress."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setTitle("Installing IRH Engine")
            self.setSubTitle("Please wait while the engine is being installed...")
            
            layout = QVBoxLayout(self)
            
            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            layout.addWidget(self.progress_bar)
            
            # Status label
            self.status_label = QLabel("Preparing installation...")
            layout.addWidget(self.status_label)
            
            # Log output
            self.log_output = QTextEdit()
            self.log_output.setReadOnly(True)
            self.log_output.setFont(QFont("Consolas", 9))
            layout.addWidget(self.log_output)
            
            self.worker = None
            self.installation_complete = False
            self.installation_success = False
        
        def initializePage(self):
            """Start installation when page is shown."""
            # Get installation parameters from previous pages
            wizard = self.wizard()
            
            if wizard.field("github_source"):
                source = "github:latest"
            elif wizard.field("bundled_source"):
                source = "bundled"
            else:
                source = wizard.field("local_path")
            
            install_path = Path(wizard.field("install_path"))
            
            # Start installation worker
            self.worker = InstallWorker(source, install_path)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        
        def on_progress(self, percent: int, message: str):
            """Handle progress updates."""
            self.progress_bar.setValue(percent)
            self.status_label.setText(message)
            self.log_output.append(f"[{percent}%] {message}")
        
        def on_finished(self, success: bool, message: str):
            """Handle installation completion."""
            self.installation_complete = True
            self.installation_success = success
            self.status_label.setText(message)
            self.log_output.append(f"\n{'✓' if success else '✗'} {message}")
            self.completeChanged.emit()
        
        def isComplete(self):
            """Check if page is complete."""
            return self.installation_complete
    
    
    class FinishPage(QWizardPage):
        """Final page of the setup wizard."""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setTitle("Setup Complete")
            self.setSubTitle("IRH Desktop is ready to use!")
            
            layout = QVBoxLayout(self)
            
            self.result_label = QLabel()
            self.result_label.setWordWrap(True)
            layout.addWidget(self.result_label)
            
            # Options
            self.launch_app = QCheckBox("Launch IRH Desktop after closing wizard")
            self.launch_app.setChecked(True)
            layout.addWidget(self.launch_app)
            
            self.view_docs = QCheckBox("Open documentation in browser")
            layout.addWidget(self.view_docs)
            
            layout.addStretch()
        
        def initializePage(self):
            """Initialize the finish page."""
            # Check if installation was successful
            progress_page = self.wizard().page(3)  # InstallProgressPage
            
            if progress_page.installation_success:
                self.result_label.setText("""
                <p><b>Installation completed successfully!</b></p>
                
                <p>The IRH computational engine has been installed and verified.</p>
                
                <p>You can now:</p>
                <ul>
                <li>Verify the Cosmic Fixed Point (Eq. 1.14)</li>
                <li>Compute the fine-structure constant α⁻¹</li>
                <li>Explore spectral dimension flow</li>
                <li>Run the full verification suite</li>
                </ul>
                
                <p>All computations will show transparent output explaining
                exactly what's happening with references to IRH21.md.</p>
                """)
            else:
                self.result_label.setText("""
                <p><b>Installation encountered issues.</b></p>
                
                <p>Please check the log for details and try again.</p>
                
                <p>You can also:</p>
                <ul>
                <li>Try a different installation source</li>
                <li>Check your internet connection</li>
                <li>Install manually from GitHub</li>
                </ul>
                """)
                self.launch_app.setChecked(False)
    
    
    class SetupWizard(QWizard):
        """
        Setup wizard for first-time IRH Desktop installation.
        
        Guides users through:
        1. Welcome and overview
        2. Installation source selection
        3. Installation location
        4. Installation progress
        5. Completion and next steps
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            
            self.setWindowTitle("IRH Desktop Setup")
            self.setMinimumSize(600, 500)
            
            # Add pages
            self.addPage(WelcomePage())
            self.addPage(InstallSourcePage())
            self.addPage(InstallLocationPage())
            self.addPage(InstallProgressPage())
            self.addPage(FinishPage())
            
            # Configure wizard
            self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
            self.setOption(QWizard.WizardOption.NoBackButtonOnLastPage, True)
        
        def done(self, result):
            """Handle wizard completion."""
            if result == QWizard.DialogCode.Accepted:
                finish_page = self.page(4)
                
                if finish_page.view_docs.isChecked():
                    import webbrowser
                    webbrowser.open("https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/docs/TECHNICAL_REFERENCE.md")
                
                if finish_page.launch_app.isChecked():
                    # Would launch main app here
                    pass
            
            super().done(result)

else:
    # Placeholder when PyQt6 is not available
    class SetupWizard:
        """Placeholder SetupWizard when PyQt6 is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyQt6 is required for IRH Desktop GUI. "
                "Install with: pip install PyQt6"
            )
        
        def show(self):
            pass
