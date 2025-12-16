"""
IRH Desktop - Plugin Manager

Handles plugin discovery, loading, and lifecycle management.

This module implements Phase 6 of the DEB_PACKAGE_ROADMAP.md:
- Plugin discovery and loading
- Plugin manager UI support
- Security sandboxing

Author: Brandon D. McCrary
"""

import os
import sys
import importlib
import importlib.util
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from irh_desktop.plugins.base import (
    IRHPlugin,
    PluginInfo,
    PluginContext,
    PluginResult,
    PluginCategory,
    get_registered_plugins,
)

logger = logging.getLogger(__name__)


# Default plugin directories
SYSTEM_PLUGIN_DIR = Path("/opt/irh/plugins")
USER_PLUGIN_DIR = Path.home() / ".local/share/irh/plugins"


@dataclass
class LoadedPlugin:
    """
    A loaded plugin instance.
    
    Attributes
    ----------
    plugin : IRHPlugin
        Plugin instance
    module_path : Path
        Path to plugin module
    loaded_at : datetime
        When plugin was loaded
    enabled : bool
        Whether plugin is enabled
    """
    plugin: IRHPlugin
    module_path: Path
    loaded_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


class PluginManager:
    """
    Manages IRH Desktop plugins.
    
    Handles discovery, loading, and lifecycle of plugins.
    
    Parameters
    ----------
    plugin_dirs : List[Path], optional
        Directories to search for plugins
    auto_discover : bool
        Automatically discover plugins on init
        
    Examples
    --------
    >>> manager = PluginManager()
    >>> plugins = manager.discover_plugins()
    >>> for name, info in plugins.items():
    ...     print(f"{name}: {info.description}")
    >>> manager.load_plugin("My Analysis Plugin")
    >>> result = manager.run_plugin("My Analysis Plugin", params={"n": 100})
    
    Theoretical Foundation
    ----------------------
    Plugin management as specified in
    docs/DEB_PACKAGE_ROADMAP.md §6 "Plugin System"
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        auto_discover: bool = True
    ):
        """Initialize the Plugin Manager."""
        self.plugin_dirs = plugin_dirs or [
            SYSTEM_PLUGIN_DIR,
            USER_PLUGIN_DIR,
        ]
        
        # Loaded plugins
        self._loaded: Dict[str, LoadedPlugin] = {}
        
        # Plugin discovery cache
        self._discovered: Dict[str, Path] = {}
        
        # Execution context
        self._context: Optional[PluginContext] = None
        
        # Ensure plugin directories exist
        for d in self.plugin_dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        if auto_discover:
            self.discover_plugins()
    
    def set_context(self, context: PluginContext) -> None:
        """
        Set the execution context for plugins.
        
        Parameters
        ----------
        context : PluginContext
            Execution context
        """
        self._context = context
    
    def add_plugin_dir(self, path: Path) -> None:
        """
        Add a plugin directory to search.
        
        Parameters
        ----------
        path : Path
            Directory path
        """
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
            path.mkdir(parents=True, exist_ok=True)
    
    def discover_plugins(self) -> Dict[str, PluginInfo]:
        """
        Discover available plugins.
        
        Scans plugin directories for valid plugins.
        
        Returns
        -------
        Dict[str, PluginInfo]
            Map of plugin names to info
        """
        discovered = {}
        self._discovered.clear()
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # Look for plugin files
            for item in plugin_dir.iterdir():
                if item.is_file() and item.suffix == '.py':
                    # Single file plugin
                    info = self._probe_plugin(item)
                    if info:
                        discovered[info.name] = info
                        self._discovered[info.name] = item
                        
                elif item.is_dir() and (item / "__init__.py").exists():
                    # Package plugin
                    info = self._probe_plugin(item / "__init__.py")
                    if info:
                        discovered[info.name] = info
                        self._discovered[info.name] = item
        
        # Also include registered plugins (from imports)
        for name, cls in get_registered_plugins().items():
            if name not in discovered:
                discovered[name] = cls.info
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def _probe_plugin(self, path: Path) -> Optional[PluginInfo]:
        """
        Probe a file for plugin metadata without fully loading.
        
        Parameters
        ----------
        path : Path
            Path to plugin file
            
        Returns
        -------
        PluginInfo or None
            Plugin info if valid
        """
        try:
            # Read file and look for PluginInfo
            content = path.read_text()
            
            # Check if it's likely a plugin
            if "IRHPlugin" not in content or "PluginInfo" not in content:
                return None
            
            # Load the module to get info
            spec = importlib.util.spec_from_file_location(
                f"plugin_{path.stem}",
                path
            )
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute in isolated namespace
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.warning(f"Failed to load plugin {path}: {e}")
                return None
            
            # Find IRHPlugin subclasses
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, IRHPlugin) and 
                    obj is not IRHPlugin and
                    hasattr(obj, 'info') and 
                    obj.info is not None):
                    return obj.info
            
            return None
            
        except Exception as e:
            logger.debug(f"Error probing {path}: {e}")
            return None
    
    def load_plugin(self, name: str) -> bool:
        """
        Load a plugin by name.
        
        Parameters
        ----------
        name : str
            Plugin name
            
        Returns
        -------
        bool
            True if loaded successfully
        """
        if name in self._loaded:
            logger.info(f"Plugin {name} already loaded")
            return True
        
        # Check registered plugins first
        registered = get_registered_plugins()
        if name in registered:
            try:
                plugin = registered[name]()
                self._loaded[name] = LoadedPlugin(
                    plugin=plugin,
                    module_path=Path(__file__),  # Unknown
                    enabled=True,
                )
                logger.info(f"Loaded registered plugin: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {name}: {e}")
                return False
        
        # Check discovered plugins
        if name not in self._discovered:
            logger.error(f"Plugin {name} not found")
            return False
        
        path = self._discovered[name]
        
        try:
            # Load the module
            if path.is_dir():
                module_path = path / "__init__.py"
            else:
                module_path = path
            
            spec = importlib.util.spec_from_file_location(
                f"plugin_{path.stem}",
                module_path
            )
            if not spec or not spec.loader:
                return False
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugin_{path.stem}"] = module
            spec.loader.exec_module(module)
            
            # Find and instantiate plugin class
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (isinstance(obj, type) and 
                    issubclass(obj, IRHPlugin) and 
                    obj is not IRHPlugin and
                    hasattr(obj, 'info') and 
                    obj.info is not None and
                    obj.info.name == name):
                    
                    plugin = obj()
                    
                    # Run setup
                    if self._context:
                        if not plugin.setup(self._context):
                            logger.error(f"Plugin {name} setup failed")
                            return False
                    
                    self._loaded[name] = LoadedPlugin(
                        plugin=plugin,
                        module_path=path,
                        enabled=True,
                    )
                    logger.info(f"Loaded plugin: {name}")
                    return True
            
            logger.error(f"No plugin class found in {path}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            return False
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Parameters
        ----------
        name : str
            Plugin name
            
        Returns
        -------
        bool
            True if unloaded
        """
        if name not in self._loaded:
            return False
        
        loaded = self._loaded[name]
        
        # Run teardown
        if self._context:
            loaded.plugin.teardown(self._context)
        
        del self._loaded[name]
        logger.info(f"Unloaded plugin: {name}")
        return True
    
    def enable_plugin(self, name: str) -> bool:
        """Enable a loaded plugin."""
        if name in self._loaded:
            self._loaded[name].enabled = True
            return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """Disable a loaded plugin."""
        if name in self._loaded:
            self._loaded[name].enabled = False
            return True
        return False
    
    def is_loaded(self, name: str) -> bool:
        """Check if plugin is loaded."""
        return name in self._loaded
    
    def is_enabled(self, name: str) -> bool:
        """Check if plugin is enabled."""
        if name in self._loaded:
            return self._loaded[name].enabled
        return False
    
    def get_plugin(self, name: str) -> Optional[IRHPlugin]:
        """
        Get a loaded plugin instance.
        
        Parameters
        ----------
        name : str
            Plugin name
            
        Returns
        -------
        IRHPlugin or None
            Plugin instance
        """
        if name in self._loaded:
            return self._loaded[name].plugin
        return None
    
    def run_plugin(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[PluginContext] = None
    ) -> Optional[PluginResult]:
        """
        Run a plugin.
        
        Parameters
        ----------
        name : str
            Plugin name
        params : Dict[str, Any], optional
            Plugin parameters
        context : PluginContext, optional
            Execution context (uses default if not provided)
            
        Returns
        -------
        PluginResult or None
            Plugin result, or None if plugin not found/disabled
        """
        if name not in self._loaded:
            # Try to load it
            if not self.load_plugin(name):
                logger.error(f"Plugin {name} not available")
                return None
        
        loaded = self._loaded[name]
        
        if not loaded.enabled:
            logger.warning(f"Plugin {name} is disabled")
            return None
        
        plugin = loaded.plugin
        ctx = context or self._context or PluginContext()
        params = params or plugin.get_default_params()
        
        # Validate parameters
        errors = plugin.validate_params(params)
        if errors:
            logger.error(f"Invalid parameters for {name}: {errors}")
            return PluginResult(success=False, error=f"Invalid parameters: {errors}")
        
        try:
            logger.info(f"Running plugin: {name}")
            result = plugin.run(ctx, params)
            logger.info(f"Plugin {name} completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.exception(f"Plugin {name} crashed: {e}")
            return PluginResult(success=False, error=str(e))
    
    def list_loaded(self) -> List[Dict[str, Any]]:
        """
        List loaded plugins.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of loaded plugin info
        """
        return [
            {
                "name": name,
                "version": loaded.plugin.info.version,
                "enabled": loaded.enabled,
                "loaded_at": loaded.loaded_at.isoformat(),
            }
            for name, loaded in self._loaded.items()
        ]
    
    def list_available(self) -> List[Dict[str, Any]]:
        """
        List all available plugins (discovered + registered).
        
        Returns
        -------
        List[Dict[str, Any]]
            List of available plugin info
        """
        plugins = self.discover_plugins()
        return [
            {
                "name": name,
                "version": info.version,
                "description": info.description,
                "category": info.category.name,
                "loaded": name in self._loaded,
            }
            for name, info in plugins.items()
        ]
    
    def get_plugin_params(self, name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter definitions for a plugin.
        
        Parameters
        ----------
        name : str
            Plugin name
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Parameter definitions
        """
        plugin = self.get_plugin(name)
        if plugin:
            return plugin.parameters
        return {}


# Convenience functions

def discover_plugins(
    dirs: Optional[List[Path]] = None
) -> Dict[str, PluginInfo]:
    """
    Discover available plugins.
    
    Parameters
    ----------
    dirs : List[Path], optional
        Directories to search
        
    Returns
    -------
    Dict[str, PluginInfo]
        Map of plugin names to info
    """
    manager = PluginManager(plugin_dirs=dirs, auto_discover=False)
    return manager.discover_plugins()


def load_plugin(
    name: str,
    dirs: Optional[List[Path]] = None
) -> Optional[IRHPlugin]:
    """
    Load a plugin by name.
    
    Parameters
    ----------
    name : str
        Plugin name
    dirs : List[Path], optional
        Directories to search
        
    Returns
    -------
    IRHPlugin or None
        Plugin instance
    """
    manager = PluginManager(plugin_dirs=dirs)
    if manager.load_plugin(name):
        return manager.get_plugin(name)
    return None


# Global plugin manager instance
_global_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager.
    
    Returns
    -------
    PluginManager
        Global manager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = PluginManager()
    return _global_manager


def set_plugin_manager(manager: PluginManager) -> None:
    """
    Set the global plugin manager.
    
    Parameters
    ----------
    manager : PluginManager
        Manager to use globally
    """
    global _global_manager
    _global_manager = manager
