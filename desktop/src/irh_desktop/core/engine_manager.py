"""
IRH Desktop - Engine Manager

Manages the IRH computational engine lifecycle including:
- Engine discovery and verification
- Installation from GitHub or local sources
- Automatic updates
- Version rollback

Theoretical Foundation:
    IRH21.md - The computational engine implements all equations
    from the Intrinsic Resonance Holography v21.0 manuscript.

Author: Brandon D. McCrary
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Default installation paths
DEFAULT_ENGINE_DIR = Path("/opt/irh/engine")
USER_ENGINE_DIR = Path.home() / ".local/share/irh/engine"
GITHUB_REPO = "brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-"
GITHUB_URL = f"https://github.com/{GITHUB_REPO}"


@dataclass
class EngineInfo:
    """
    Information about an installed IRH engine.
    
    Attributes
    ----------
    path : Path
        Installation directory
    version : str
        Engine version (e.g., "21.0.0")
    commit : str
        Git commit hash
    installed_date : datetime
        When the engine was installed
    verified : bool
        Whether the engine passed verification
    python_version : str
        Python version for this engine
    """
    path: Path
    version: str = "unknown"
    commit: str = "unknown"
    installed_date: datetime = field(default_factory=datetime.now)
    verified: bool = False
    python_version: str = ""
    
    def __post_init__(self):
        if not self.python_version:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"


@dataclass
class UpdateInfo:
    """
    Information about available updates.
    
    Attributes
    ----------
    available : bool
        Whether an update is available
    current_version : str
        Currently installed version
    latest_version : str
        Latest available version
    changelog : List[str]
        List of changes in the update
    download_url : str
        URL to download the update
    """
    available: bool
    current_version: str
    latest_version: str
    changelog: List[str] = field(default_factory=list)
    download_url: str = ""
    commit_hash: str = ""


@dataclass
class VerificationResult:
    """
    Result of engine verification.
    
    Attributes
    ----------
    success : bool
        Whether verification passed
    tests_passed : int
        Number of tests passed
    tests_failed : int
        Number of tests failed
    errors : List[str]
        Error messages
    details : Dict[str, Any]
        Additional verification details
    """
    success: bool
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class EngineManager:
    """
    Manages IRH computational engine lifecycle.
    
    This class handles discovery, installation, updates, and verification
    of the IRH computational engine.
    
    Parameters
    ----------
    install_dir : Path, optional
        Directory where engines are installed
        
    Examples
    --------
    >>> manager = EngineManager()
    >>> engines = manager.discover_engines()
    >>> if not engines:
    ...     manager.install_engine("github:latest")
    >>> update = manager.check_for_updates()
    >>> if update.available:
    ...     manager.install_update(update)
    
    Theoretical Foundation
    ----------------------
    The IRH engine implements the complete mathematical formalism from
    IRH21.md, including:
    - Quaternionic Group Field Theory (§1)
    - Renormalization Group Flow (§1.2-1.3)
    - Emergent Spacetime (§2)
    - Standard Model Derivation (§3)
    """
    
    def __init__(self, install_dir: Optional[Path] = None):
        """
        Initialize the Engine Manager.
        
        Parameters
        ----------
        install_dir : Path, optional
            Custom installation directory. If not provided, uses
            /opt/irh/engine (system) or ~/.local/share/irh/engine (user)
        """
        self.install_dir = install_dir or self._get_default_install_dir()
        self._current_engine: Optional[EngineInfo] = None
        
    def _get_default_install_dir(self) -> Path:
        """Get the default installation directory."""
        # Try system directory first, fall back to user directory
        if DEFAULT_ENGINE_DIR.exists() or os.access(DEFAULT_ENGINE_DIR.parent, os.W_OK):
            return DEFAULT_ENGINE_DIR
        return USER_ENGINE_DIR
    
    def discover_engines(self) -> List[EngineInfo]:
        """
        Discover installed IRH engines.
        
        Searches known locations for IRH engine installations.
        
        Returns
        -------
        List[EngineInfo]
            List of discovered engine installations
        """
        engines = []
        
        search_paths = [
            DEFAULT_ENGINE_DIR,
            USER_ENGINE_DIR,
            self.install_dir,
            Path.cwd(),  # Current directory might be the repo
        ]
        
        for path in set(search_paths):  # Deduplicate
            if self._is_valid_engine(path):
                info = self._get_engine_info(path)
                if info:
                    engines.append(info)
        
        return engines
    
    def _is_valid_engine(self, path: Path) -> bool:
        """
        Check if a path contains a valid IRH engine.
        
        Parameters
        ----------
        path : Path
            Directory to check
            
        Returns
        -------
        bool
            True if path contains valid engine
        """
        if not path.exists():
            return False
        
        # Check for key IRH files
        required_files = [
            "IRH21.md",  # Theoretical manuscript
            "src/primitives/quaternions.py",  # Core primitives
            "src/rg_flow/beta_functions.py",  # RG flow
        ]
        
        for rel_path in required_files:
            if not (path / rel_path).exists():
                return False
        
        return True
    
    def _get_engine_info(self, path: Path) -> Optional[EngineInfo]:
        """
        Get information about an installed engine.
        
        Parameters
        ----------
        path : Path
            Engine installation directory
            
        Returns
        -------
        EngineInfo or None
            Engine information, or None if not valid
        """
        try:
            # Get version from pyproject.toml or __init__.py
            version = self._extract_version(path)
            
            # Get git commit if available
            commit = self._get_git_commit(path)
            
            # Get installed date from file modification time
            irh21_path = path / "IRH21.md"
            if irh21_path.exists():
                mtime = irh21_path.stat().st_mtime
                installed_date = datetime.fromtimestamp(mtime)
            else:
                installed_date = datetime.now()
            
            return EngineInfo(
                path=path,
                version=version,
                commit=commit,
                installed_date=installed_date,
                verified=False
            )
        except Exception as e:
            logger.error(f"Error getting engine info from {path}: {e}")
            return None
    
    def _extract_version(self, path: Path) -> str:
        """Extract version from engine files."""
        # Try pyproject.toml
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                for line in content.splitlines():
                    if line.strip().startswith("version"):
                        # Parse version = "X.Y.Z"
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"\'')
            except Exception:
                pass
        
        # Try __init__.py
        init_file = path / "src" / "__init__.py"
        if init_file.exists():
            try:
                content = init_file.read_text()
                for line in content.splitlines():
                    if "__version__" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"\'')
            except Exception:
                pass
        
        return "21.0.0"  # Default version
    
    def _get_git_commit(self, path: Path) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
        except Exception:
            pass
        return "unknown"
    
    def install_engine(
        self,
        source: str = "github:latest",
        path: Optional[Path] = None,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Install the IRH engine.
        
        Parameters
        ----------
        source : str
            Installation source:
            - "github:latest" - Clone latest from GitHub
            - "github:tag" - Clone specific tag
            - "/path/to/repo" - Copy from local path
        path : Path, optional
            Installation directory (default: self.install_dir)
        progress_callback : callable, optional
            Callback for progress updates: callback(percent, message)
            
        Returns
        -------
        bool
            True if installation succeeded
        """
        install_path = path or self.install_dir
        
        try:
            if progress_callback:
                progress_callback(0, "Starting installation...")
            
            # Create installation directory
            install_path.mkdir(parents=True, exist_ok=True)
            
            if source.startswith("github:"):
                # Clone from GitHub
                tag = source.split(":", 1)[1]
                success = self._clone_from_github(install_path, tag, progress_callback)
            elif Path(source).exists():
                # Copy from local path
                success = self._copy_from_local(Path(source), install_path, progress_callback)
            else:
                logger.error(f"Unknown source: {source}")
                return False
            
            if success:
                if progress_callback:
                    progress_callback(90, "Verifying installation...")
                
                # Verify the installation
                result = self.verify_engine(EngineInfo(path=install_path))
                
                if progress_callback:
                    progress_callback(100, "Installation complete!")
                
                logger.info(f"Engine installed successfully at {install_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            return False
    
    def _clone_from_github(
        self,
        path: Path,
        tag: str,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Clone repository from GitHub."""
        try:
            if progress_callback:
                progress_callback(10, f"Cloning from GitHub ({tag})...")
            
            cmd = ["git", "clone", "--depth", "1"]
            
            if tag != "latest":
                cmd.extend(["--branch", tag])
            
            cmd.extend([GITHUB_URL + ".git", str(path)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Git clone failed: {result.stderr}")
                return False
            
            if progress_callback:
                progress_callback(70, "Installing dependencies...")
            
            # Install Python dependencies
            pip_cmd = [
                sys.executable, "-m", "pip", "install", "-e", str(path),
                "--quiet"
            ]
            subprocess.run(pip_cmd, capture_output=True, timeout=300)
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Git clone timed out")
            return False
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            return False
    
    def _copy_from_local(
        self,
        source: Path,
        dest: Path,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Copy engine from local path."""
        try:
            if progress_callback:
                progress_callback(10, f"Copying from {source}...")
            
            # Remove existing if present
            if dest.exists():
                shutil.rmtree(dest)
            
            # Copy directory
            shutil.copytree(source, dest)
            
            if progress_callback:
                progress_callback(70, "Installation copied")
            
            return True
            
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return False
    
    def check_for_updates(self) -> UpdateInfo:
        """
        Check for available updates.
        
        Returns
        -------
        UpdateInfo
            Information about available updates
        """
        # Get current engine info
        engines = self.discover_engines()
        current_version = engines[0].version if engines else "0.0.0"
        current_commit = engines[0].commit if engines else "unknown"
        
        try:
            # Check GitHub for latest
            import requests
            
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                latest_commit = data["sha"][:8]
                
                # Check if update available
                available = latest_commit != current_commit
                
                # Get recent commits for changelog
                changelog = []
                if available:
                    commits_url = f"https://api.github.com/repos/{GITHUB_REPO}/commits"
                    commits_response = requests.get(commits_url, timeout=10)
                    if commits_response.status_code == 200:
                        commits = commits_response.json()[:10]
                        changelog = [c["commit"]["message"].split("\n")[0] for c in commits]
                
                return UpdateInfo(
                    available=available,
                    current_version=current_version,
                    latest_version="21.0.0",  # Would parse from repo
                    changelog=changelog,
                    download_url=GITHUB_URL,
                    commit_hash=latest_commit
                )
            
        except Exception as e:
            logger.warning(f"Update check failed: {e}")
        
        return UpdateInfo(
            available=False,
            current_version=current_version,
            latest_version=current_version
        )
    
    def install_update(self, update_info: UpdateInfo) -> bool:
        """
        Install an available update.
        
        Parameters
        ----------
        update_info : UpdateInfo
            Update information from check_for_updates()
            
        Returns
        -------
        bool
            True if update succeeded
        """
        if not update_info.available:
            logger.info("No update available")
            return True
        
        # Backup current installation
        engines = self.discover_engines()
        if engines:
            backup_path = engines[0].path.parent / f"engine_backup_{datetime.now():%Y%m%d_%H%M%S}"
            try:
                shutil.copytree(engines[0].path, backup_path)
                logger.info(f"Backed up current installation to {backup_path}")
            except Exception as e:
                logger.warning(f"Backup failed: {e}")
        
        # Pull latest changes
        if engines and (engines[0].path / ".git").exists():
            try:
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=engines[0].path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    logger.info("Update installed successfully")
                    return True
                else:
                    logger.error(f"Git pull failed: {result.stderr}")
            except Exception as e:
                logger.error(f"Update failed: {e}")
        
        return False
    
    def verify_engine(self, engine: EngineInfo) -> VerificationResult:
        """
        Verify an engine installation.
        
        Runs basic verification tests to ensure the engine is
        properly installed and functional.
        
        Parameters
        ----------
        engine : EngineInfo
            Engine to verify
            
        Returns
        -------
        VerificationResult
            Verification results
        """
        errors = []
        tests_passed = 0
        tests_failed = 0
        details = {}
        
        # Test 1: Check required files exist
        required_files = [
            "IRH21.md",
            "src/primitives/quaternions.py",
            "src/rg_flow/beta_functions.py",
            "src/rg_flow/fixed_points.py",
        ]
        
        for rel_path in required_files:
            full_path = engine.path / rel_path
            if full_path.exists():
                tests_passed += 1
            else:
                tests_failed += 1
                errors.append(f"Missing required file: {rel_path}")
        
        # Test 2: Try importing core modules
        sys.path.insert(0, str(engine.path))
        try:
            from src.primitives.quaternions import Quaternion
            tests_passed += 1
            details["quaternions"] = "importable"
        except Exception as e:
            tests_failed += 1
            errors.append(f"Cannot import quaternions: {e}")
        
        try:
            from src.rg_flow.fixed_points import find_fixed_point
            tests_passed += 1
            details["fixed_points"] = "importable"
        except Exception as e:
            tests_failed += 1
            errors.append(f"Cannot import fixed_points: {e}")
        
        # Test 3: Quick functional test
        try:
            from src.rg_flow.fixed_points import find_fixed_point
            fp = find_fixed_point()
            if hasattr(fp, 'lambda_star'):
                tests_passed += 1
                details["fixed_point_computed"] = True
            else:
                tests_failed += 1
                errors.append("Fixed point computation returned unexpected result")
        except Exception as e:
            tests_failed += 1
            errors.append(f"Fixed point computation failed: {e}")
        finally:
            sys.path.pop(0)
        
        success = tests_failed == 0
        engine.verified = success
        
        return VerificationResult(
            success=success,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            errors=errors,
            details=details
        )
    
    def rollback_engine(self, backup_path: Path) -> bool:
        """
        Rollback to a previous engine version.
        
        Parameters
        ----------
        backup_path : Path
            Path to backup to restore
            
        Returns
        -------
        bool
            True if rollback succeeded
        """
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            # Remove current
            if self.install_dir.exists():
                shutil.rmtree(self.install_dir)
            
            # Restore backup
            shutil.copytree(backup_path, self.install_dir)
            
            logger.info(f"Rolled back to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_current_engine(self) -> Optional[EngineInfo]:
        """
        Get the currently active engine.
        
        Returns
        -------
        EngineInfo or None
            Current engine, or None if none installed
        """
        engines = self.discover_engines()
        return engines[0] if engines else None
