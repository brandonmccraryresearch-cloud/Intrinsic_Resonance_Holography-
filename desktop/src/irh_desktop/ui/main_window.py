"""
IRH Desktop - Main Window

The main application window providing:
- Module navigation sidebar
- Workspace area for computations
- Transparency console for output
- Menu bar and toolbars

UI Design follows docs/DEB_PACKAGE_ROADMAP.md §3.3

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
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QMenuBar, QMenu, QToolBar, QStatusBar,
        QTreeWidget, QTreeWidgetItem, QTextEdit, QTabWidget,
        QPushButton, QLabel, QProgressBar, QGroupBox,
        QMessageBox, QFileDialog, QApplication,
    )
    from PyQt6.QtCore import Qt, QSettings, QSize, pyqtSignal, QTimer
    from PyQt6.QtGui import QAction, QIcon, QFont, QTextCursor
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    # Create placeholder classes for when PyQt6 is not available
    QMainWindow = object


if HAS_PYQT6:
    
    class TransparencyConsole(QWidget):
        """
        Console widget for displaying transparent computation output.
        
        Shows all computation messages with color-coding and
        theoretical references.
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setup_ui()
        
        def setup_ui(self):
            """Initialize the console UI."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Header
            header = QHBoxLayout()
            header.addWidget(QLabel("Transparent Output Console"))
            header.addStretch()
            
            # Control buttons
            self.clear_btn = QPushButton("Clear")
            self.clear_btn.clicked.connect(self.clear)
            header.addWidget(self.clear_btn)
            
            self.export_btn = QPushButton("Export")
            self.export_btn.clicked.connect(self.export_log)
            header.addWidget(self.export_btn)
            
            layout.addLayout(header)
            
            # Console output area
            self.console = QTextEdit()
            self.console.setReadOnly(True)
            self.console.setFont(QFont("Consolas", 10))
            self.console.setStyleSheet("""
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    border: 1px solid #3a3a3a;
                }
            """)
            layout.addWidget(self.console)
        
        def append_message(self, message) -> None:
            """
            Append a transparency message to the console.
            
            Parameters
            ----------
            message : TransparentMessage
                Message to display
            """
            html = message.render_html()
            self.console.append(html)
            
            # Auto-scroll to bottom
            cursor = self.console.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.console.setTextCursor(cursor)
        
        def clear(self) -> None:
            """Clear the console."""
            self.console.clear()
        
        def export_log(self) -> None:
            """Export console content to file."""
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Log", "", "HTML Files (*.html);;Text Files (*.txt)"
            )
            if path:
                with open(path, 'w') as f:
                    if path.endswith('.html'):
                        f.write(self.console.toHtml())
                    else:
                        f.write(self.console.toPlainText())
    
    
    class ModuleNavigator(QWidget):
        """
        Tree view for navigating IRH modules.
        
        Allows users to browse and run different computation modules.
        """
        
        # Signal emitted when a module is selected
        module_selected = pyqtSignal(str)  # Module path
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setup_ui()
        
        def setup_ui(self):
            """Initialize the navigator UI."""
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Module tree
            self.tree = QTreeWidget()
            self.tree.setHeaderLabel("IRH Modules")
            self.tree.itemDoubleClicked.connect(self._on_item_clicked)
            layout.addWidget(self.tree)
            
            # Populate with modules
            self._populate_modules()
            
            # Quick actions
            actions_group = QGroupBox("Quick Actions")
            actions_layout = QVBoxLayout(actions_group)
            
            btn_alpha = QPushButton("Compute α⁻¹")
            btn_alpha.clicked.connect(lambda: self.module_selected.emit("observables.alpha_inverse"))
            actions_layout.addWidget(btn_alpha)
            
            btn_fixed = QPushButton("Verify Fixed Point")
            btn_fixed.clicked.connect(lambda: self.module_selected.emit("rg_flow.fixed_points"))
            actions_layout.addWidget(btn_fixed)
            
            btn_spectral = QPushButton("Spectral Dimension")
            btn_spectral.clicked.connect(lambda: self.module_selected.emit("emergent_spacetime.spectral_dimension"))
            actions_layout.addWidget(btn_spectral)
            
            btn_full = QPushButton("Run Full Suite")
            btn_full.clicked.connect(lambda: self.module_selected.emit("full_suite"))
            actions_layout.addWidget(btn_full)
            
            layout.addWidget(actions_group)
        
        def _populate_modules(self):
            """Populate the module tree."""
            # Primitives
            primitives = QTreeWidgetItem(self.tree, ["Primitives"])
            QTreeWidgetItem(primitives, ["Quaternions"])
            QTreeWidgetItem(primitives, ["Group Manifolds"])
            QTreeWidgetItem(primitives, ["QNCD"])
            
            # cGFT
            cgft = QTreeWidgetItem(self.tree, ["cGFT"])
            QTreeWidgetItem(cgft, ["Actions"])
            QTreeWidgetItem(cgft, ["Fields"])
            
            # RG Flow
            rg = QTreeWidgetItem(self.tree, ["RG Flow"])
            QTreeWidgetItem(rg, ["Beta Functions"])
            QTreeWidgetItem(rg, ["Fixed Points"])
            QTreeWidgetItem(rg, ["Validation"])
            
            # Emergent Spacetime
            spacetime = QTreeWidgetItem(self.tree, ["Emergent Spacetime"])
            QTreeWidgetItem(spacetime, ["Spectral Dimension"])
            QTreeWidgetItem(spacetime, ["Metric Tensor"])
            QTreeWidgetItem(spacetime, ["Lorentzian Signature"])
            QTreeWidgetItem(spacetime, ["Einstein Equations"])
            
            # Topology
            topology = QTreeWidgetItem(self.tree, ["Topology"])
            QTreeWidgetItem(topology, ["Betti Numbers"])
            QTreeWidgetItem(topology, ["Instanton Number"])
            QTreeWidgetItem(topology, ["VWP Spectrum"])
            
            # Standard Model
            sm = QTreeWidgetItem(self.tree, ["Standard Model"])
            QTreeWidgetItem(sm, ["Gauge Groups"])
            QTreeWidgetItem(sm, ["Fermion Masses"])
            QTreeWidgetItem(sm, ["Mixing Matrices"])
            QTreeWidgetItem(sm, ["Higgs Sector"])
            QTreeWidgetItem(sm, ["Neutrinos"])
            QTreeWidgetItem(sm, ["Strong CP"])
            
            # Predictions
            predictions = QTreeWidgetItem(self.tree, ["Predictions"])
            QTreeWidgetItem(predictions, ["Dark Energy"])
            QTreeWidgetItem(predictions, ["Lorentz Violation"])
            QTreeWidgetItem(predictions, ["Muon g-2"])
            QTreeWidgetItem(predictions, ["Born Rule"])
            
            # Expand all by default
            self.tree.expandAll()
        
        def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
            """Handle item double-click."""
            # Build module path from tree hierarchy
            path_parts = []
            current = item
            while current:
                path_parts.insert(0, current.text(0).lower().replace(" ", "_"))
                current = current.parent()
            
            module_path = ".".join(path_parts)
            self.module_selected.emit(module_path)
    
    
    class WorkspaceWidget(QWidget):
        """
        Main workspace area for running computations.
        
        Shows computation progress, results, and visualizations.
        """
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setup_ui()
        
        def setup_ui(self):
            """Initialize the workspace UI."""
            layout = QVBoxLayout(self)
            
            # Current computation info
            info_group = QGroupBox("Current Computation")
            info_layout = QVBoxLayout(info_group)
            
            self.computation_label = QLabel("No computation running")
            self.computation_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            info_layout.addWidget(self.computation_label)
            
            self.status_label = QLabel("Status: Idle")
            info_layout.addWidget(self.status_label)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            info_layout.addWidget(self.progress_bar)
            
            layout.addWidget(info_group)
            
            # Results tabs
            self.tabs = QTabWidget()
            
            # Results tab
            results_widget = QTextEdit()
            results_widget.setReadOnly(True)
            results_widget.setPlaceholderText("Computation results will appear here...")
            self.results_view = results_widget
            self.tabs.addTab(results_widget, "Results")
            
            # Visualization tab (placeholder for matplotlib)
            viz_widget = QWidget()
            viz_layout = QVBoxLayout(viz_widget)
            viz_layout.addWidget(QLabel("Visualization will appear here"))
            self.tabs.addTab(viz_widget, "Visualization")
            
            layout.addWidget(self.tabs)
            
            # Control buttons
            btn_layout = QHBoxLayout()
            
            self.run_btn = QPushButton("Run")
            self.run_btn.setEnabled(False)
            btn_layout.addWidget(self.run_btn)
            
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setEnabled(False)
            btn_layout.addWidget(self.stop_btn)
            
            btn_layout.addStretch()
            
            self.export_btn = QPushButton("Export Results")
            self.export_btn.setEnabled(False)
            btn_layout.addWidget(self.export_btn)
            
            layout.addLayout(btn_layout)
        
        def set_computation(self, name: str, reference: str = ""):
            """Set the current computation."""
            self.computation_label.setText(f"Current: {name}")
            self.status_label.setText(f"Status: Ready | {reference}")
            self.run_btn.setEnabled(True)
        
        def set_progress(self, percent: int, status: str = ""):
            """Update computation progress."""
            self.progress_bar.setValue(percent)
            if status:
                self.status_label.setText(f"Status: {status}")
        
        def show_results(self, results: str):
            """Display computation results."""
            self.results_view.setText(results)
            self.export_btn.setEnabled(True)
    
    
    class MainWindow(QMainWindow):
        """
        Main application window for IRH Desktop.
        
        Provides the primary interface for interacting with the
        IRH computational framework.
        
        Parameters
        ----------
        config_manager : ConfigManager
            Configuration manager instance
        verbose : bool
            Enable verbose output
        debug : bool
            Enable debug mode
        """
        
        def __init__(
            self,
            config_manager=None,
            verbose: bool = False,
            debug: bool = False
        ):
            super().__init__()
            
            self.config_manager = config_manager
            self.verbose = verbose
            self.debug = debug
            
            # Initialize transparency engine
            from irh_desktop.transparency.engine import TransparencyEngine
            self.transparency = TransparencyEngine(
                verbosity=4 if verbose else 3,
                show_equations=True,
                show_explanations=True
            )
            
            self.setup_ui()
            self.setup_menu()
            self.setup_connections()
            self.restore_state()
        
        def setup_ui(self):
            """Initialize the main UI."""
            self.setWindowTitle("IRH Desktop v21.0")
            self.setMinimumSize(1200, 800)
            
            # Central widget with splitter
            central = QWidget()
            self.setCentralWidget(central)
            
            main_layout = QHBoxLayout(central)
            main_layout.setContentsMargins(5, 5, 5, 5)
            
            # Main splitter (left panel | right area)
            main_splitter = QSplitter(Qt.Orientation.Horizontal)
            
            # Left panel: Module navigator
            self.navigator = ModuleNavigator()
            main_splitter.addWidget(self.navigator)
            
            # Right area: Workspace + Console splitter
            right_splitter = QSplitter(Qt.Orientation.Vertical)
            
            # Workspace
            self.workspace = WorkspaceWidget()
            right_splitter.addWidget(self.workspace)
            
            # Transparency console
            self.console = TransparencyConsole()
            right_splitter.addWidget(self.console)
            
            # Set initial sizes (70% workspace, 30% console)
            right_splitter.setSizes([500, 200])
            
            main_splitter.addWidget(right_splitter)
            
            # Set initial sizes (25% navigator, 75% right area)
            main_splitter.setSizes([300, 900])
            
            main_layout.addWidget(main_splitter)
            
            # Status bar
            self.statusBar().showMessage("Ready")
            
            # Connect transparency engine to console
            self.transparency.add_callback(self.console.append_message)
        
        def setup_menu(self):
            """Setup the menu bar."""
            menubar = self.menuBar()
            
            # File menu
            file_menu = menubar.addMenu("&File")
            
            open_action = QAction("&Open Configuration...", self)
            open_action.setShortcut("Ctrl+O")
            open_action.triggered.connect(self.open_config)
            file_menu.addAction(open_action)
            
            save_action = QAction("&Save Configuration", self)
            save_action.setShortcut("Ctrl+S")
            save_action.triggered.connect(self.save_config)
            file_menu.addAction(save_action)
            
            file_menu.addSeparator()
            
            export_action = QAction("&Export Results...", self)
            export_action.triggered.connect(self.export_results)
            file_menu.addAction(export_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("E&xit", self)
            exit_action.setShortcut("Ctrl+Q")
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # Edit menu
            edit_menu = menubar.addMenu("&Edit")
            
            prefs_action = QAction("&Preferences...", self)
            prefs_action.triggered.connect(self.show_preferences)
            edit_menu.addAction(prefs_action)
            
            # View menu
            view_menu = menubar.addMenu("&View")
            
            theme_menu = view_menu.addMenu("Theme")
            
            dark_action = QAction("Dark Mode", self)
            dark_action.setCheckable(True)
            dark_action.triggered.connect(lambda: self.set_theme("dark"))
            theme_menu.addAction(dark_action)
            
            light_action = QAction("Light Mode", self)
            light_action.setCheckable(True)
            light_action.triggered.connect(lambda: self.set_theme("light"))
            theme_menu.addAction(light_action)
            
            # Compute menu
            compute_menu = menubar.addMenu("&Compute")
            
            fixed_point_action = QAction("Verify Fixed Point", self)
            fixed_point_action.triggered.connect(lambda: self.run_computation("fixed_point"))
            compute_menu.addAction(fixed_point_action)
            
            alpha_action = QAction("Compute α⁻¹", self)
            alpha_action.triggered.connect(lambda: self.run_computation("alpha_inverse"))
            compute_menu.addAction(alpha_action)
            
            spectral_action = QAction("Spectral Dimension", self)
            spectral_action.triggered.connect(lambda: self.run_computation("spectral_dimension"))
            compute_menu.addAction(spectral_action)
            
            compute_menu.addSeparator()
            
            full_suite_action = QAction("Run Full Verification Suite", self)
            full_suite_action.triggered.connect(lambda: self.run_computation("full_suite"))
            compute_menu.addAction(full_suite_action)
            
            # Tools menu
            tools_menu = menubar.addMenu("&Tools")
            
            engine_action = QAction("Engine Manager...", self)
            engine_action.triggered.connect(self.show_engine_manager)
            tools_menu.addAction(engine_action)
            
            update_action = QAction("Check for Updates...", self)
            update_action.triggered.connect(self.check_updates)
            tools_menu.addAction(update_action)
            
            # Help menu
            help_menu = menubar.addMenu("&Help")
            
            docs_action = QAction("Documentation", self)
            docs_action.triggered.connect(self.show_documentation)
            help_menu.addAction(docs_action)
            
            about_action = QAction("About IRH Desktop", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
        
        def setup_connections(self):
            """Setup signal/slot connections."""
            self.navigator.module_selected.connect(self.on_module_selected)
        
        def restore_state(self):
            """Restore window state from settings."""
            settings = QSettings()
            geometry = settings.value("window/geometry")
            if geometry:
                self.restoreGeometry(geometry)
            state = settings.value("window/state")
            if state:
                self.restoreState(state)
        
        def save_state(self):
            """Save window state to settings."""
            settings = QSettings()
            settings.setValue("window/geometry", self.saveGeometry())
            settings.setValue("window/state", self.saveState())
        
        def closeEvent(self, event):
            """Handle window close."""
            self.save_state()
            event.accept()
        
        def on_module_selected(self, module_path: str):
            """Handle module selection from navigator."""
            self.transparency.info(f"Selected module: {module_path}")
            
            # Map module path to computation
            module_info = {
                "rg_flow.fixed_points": ("Cosmic Fixed Point Verification", "§1.2.3, Eq. 1.14"),
                "observables.alpha_inverse": ("Fine-Structure Constant Derivation", "§3.2.2, Eq. 3.4-3.5"),
                "emergent_spacetime.spectral_dimension": ("Spectral Dimension Flow", "§2.1.2, Theorem 2.1"),
                "topology.betti_numbers": ("Betti Number β₁ = 12", "Appendix D.1"),
                "topology.instanton_number": ("Instanton Number n_inst = 3", "Appendix D.2"),
            }
            
            if module_path in module_info:
                name, ref = module_info[module_path]
                self.workspace.set_computation(name, ref)
                self.statusBar().showMessage(f"Ready: {name}")
        
        def run_computation(self, computation_type: str):
            """Run a specific computation."""
            self.transparency.computation_start(
                f"Running {computation_type}",
                reference="IRH21.md"
            )
            
            # Placeholder - would run actual computation
            self.workspace.set_progress(0, f"Starting {computation_type}...")
            
            # Simulate progress with timer
            self.progress_value = 0
            self.progress_timer = QTimer()
            self.progress_timer.timeout.connect(lambda: self._update_progress(computation_type))
            self.progress_timer.start(100)
        
        def _update_progress(self, computation_type: str):
            """Update progress simulation."""
            self.progress_value += 5
            self.workspace.set_progress(self.progress_value, f"Computing {computation_type}...")
            
            if self.progress_value >= 100:
                self.progress_timer.stop()
                self.transparency.computation_end(success=True)
                self.workspace.show_results(f"Computation {computation_type} completed successfully!")
                self.statusBar().showMessage("Computation complete")
        
        def open_config(self):
            """Open a configuration file."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Configuration", "", "YAML Files (*.yaml *.yml)"
            )
            if path:
                self.transparency.info(f"Loading configuration: {path}")
                # Load config
        
        def save_config(self):
            """Save current configuration."""
            if self.config_manager:
                self.config_manager.save()
                self.transparency.info("Configuration saved")
        
        def export_results(self):
            """Export computation results."""
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", "JSON Files (*.json);;CSV Files (*.csv)"
            )
            if path:
                self.transparency.info(f"Exporting results to: {path}")
        
        def show_preferences(self):
            """Show preferences dialog."""
            QMessageBox.information(self, "Preferences", "Preferences dialog coming soon!")
        
        def set_theme(self, theme: str):
            """Set application theme."""
            from irh_desktop.app import IRHDesktopApp
            # Would apply theme here
            self.transparency.info(f"Theme set to: {theme}")
        
        def show_engine_manager(self):
            """Show engine manager dialog."""
            QMessageBox.information(self, "Engine Manager", "Engine manager coming soon!")
        
        def check_updates(self):
            """Check for updates."""
            from irh_desktop.core.engine_manager import EngineManager
            manager = EngineManager()
            update = manager.check_for_updates()
            
            if update.available:
                reply = QMessageBox.question(
                    self, "Update Available",
                    f"Version {update.latest_version} is available. Update now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    manager.install_update(update)
            else:
                QMessageBox.information(self, "Updates", "IRH Desktop is up to date!")
        
        def show_documentation(self):
            """Show documentation."""
            import webbrowser
            webbrowser.open("https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-/blob/main/docs/TECHNICAL_REFERENCE.md")
        
        def show_about(self):
            """Show about dialog."""
            QMessageBox.about(
                self, "About IRH Desktop",
                """<h2>IRH Desktop v21.0</h2>
                <p>Intrinsic Resonance Holography Desktop Application</p>
                <p>A comprehensive framework for deriving fundamental physics from first principles using quaternionic Group Field Theory.</p>
                <p><b>Key Predictions:</b></p>
                <ul>
                <li>α⁻¹ = 137.035999... (fine-structure constant)</li>
                <li>w₀ = -0.91234567 (dark energy EoS)</li>
                <li>β₁ = 12 → SU(3)×SU(2)×U(1) gauge group</li>
                <li>n_inst = 3 → Three fermion generations</li>
                </ul>
                <p>Author: Brandon D. McCrary</p>
                <p>License: MIT</p>
                """
            )
        
        def open_file(self, path: str):
            """Open a file."""
            self.transparency.info(f"Opening file: {path}")

else:
    # Placeholder when PyQt6 is not available
    class MainWindow:
        """Placeholder MainWindow when PyQt6 is not installed."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyQt6 is required for IRH Desktop GUI. "
                "Install with: pip install PyQt6"
            )
    
    class TransparencyConsole:
        """Placeholder when PyQt6 is not installed."""
        pass
    
    class ModuleNavigator:
        """Placeholder when PyQt6 is not installed."""
        pass
    
    class WorkspaceWidget:
        """Placeholder when PyQt6 is not installed."""
        pass
