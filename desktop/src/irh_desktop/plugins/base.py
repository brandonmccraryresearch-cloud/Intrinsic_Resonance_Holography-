"""
IRH Desktop - Plugin Base Classes

Defines the plugin interface and base classes for creating
IRH Desktop extensions.

This module implements Phase 6 of the DEB_PACKAGE_ROADMAP.md:
- Plugin API documentation
- Plugin base class

Author: Brandon D. McCrary
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Type
from enum import Enum, auto

logger = logging.getLogger(__name__)


class PluginCategory(Enum):
    """Categories of plugins."""
    COMPUTATION = auto()   # Adds new computation types
    VISUALIZATION = auto() # Adds new visualizations
    EXPORT = auto()        # Adds new export formats
    ANALYSIS = auto()      # Adds analysis tools
    UTILITY = auto()       # General utilities
    INTEGRATION = auto()   # External integrations


@dataclass
class PluginInfo:
    """
    Metadata about a plugin.
    
    Attributes
    ----------
    name : str
        Plugin name
    version : str
        Plugin version
    author : str
        Plugin author
    description : str
        Short description
    category : PluginCategory
        Plugin category
    requires : List[str]
        Required dependencies
    homepage : str
        Plugin homepage URL
    """
    name: str
    version: str = "1.0.0"
    author: str = ""
    description: str = ""
    category: PluginCategory = PluginCategory.UTILITY
    requires: List[str] = field(default_factory=list)
    homepage: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category.name,
            "requires": self.requires,
            "homepage": self.homepage,
        }


@dataclass
class PluginContext:
    """
    Context passed to plugins during execution.
    
    Provides access to IRH Desktop services and data.
    
    Attributes
    ----------
    engine_path : str
        Path to IRH engine
    data_dir : str
        Plugin data directory
    config : Dict[str, Any]
        Plugin configuration
    transparency_callback : callable
        Function to emit transparency messages
    progress_callback : callable
        Function to report progress
    """
    engine_path: str = ""
    data_dir: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    transparency_callback: Optional[Callable[[str, str], None]] = None
    progress_callback: Optional[Callable[[int, str], None]] = None
    
    def log_info(self, message: str, reference: str = "") -> None:
        """Log an info message through transparency engine."""
        if self.transparency_callback:
            self.transparency_callback("INFO", f"{message} | {reference}" if reference else message)
        logger.info(message)
    
    def log_step(self, message: str) -> None:
        """Log a computation step."""
        if self.transparency_callback:
            self.transparency_callback("STEP", message)
        logger.info(f"STEP: {message}")
    
    def log_error(self, message: str) -> None:
        """Log an error."""
        if self.transparency_callback:
            self.transparency_callback("ERROR", message)
        logger.error(message)
    
    def report_progress(self, percent: int, message: str = "") -> None:
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent, message)


@dataclass
class PluginResult:
    """
    Result returned from plugin execution.
    
    Attributes
    ----------
    success : bool
        Whether execution succeeded
    data : Dict[str, Any]
        Result data
    error : str
        Error message if failed
    artifacts : List[str]
        Paths to generated files
    """
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    artifacts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "artifacts": self.artifacts,
        }


class IRHPlugin(ABC):
    """
    Base class for IRH Desktop plugins.
    
    Plugins must inherit from this class and implement the
    required abstract methods.
    
    Class Attributes
    ----------------
    info : PluginInfo
        Plugin metadata (must be defined in subclass)
    parameters : Dict[str, Dict]
        Parameter definitions for GUI
        
    Examples
    --------
    >>> class MyPlugin(IRHPlugin):
    ...     info = PluginInfo(
    ...         name="My Plugin",
    ...         version="1.0.0",
    ...         author="Your Name",
    ...         description="Does something useful",
    ...         category=PluginCategory.ANALYSIS,
    ...     )
    ...     
    ...     parameters = {
    ...         "threshold": {
    ...             "type": "float",
    ...             "default": 0.1,
    ...             "min": 0.0,
    ...             "max": 1.0,
    ...             "description": "Analysis threshold",
    ...         },
    ...     }
    ...     
    ...     def run(self, context, params):
    ...         context.log_info("Starting analysis...")
    ...         # Do analysis
    ...         return PluginResult(success=True, data={"result": 42})
    
    Theoretical Foundation
    ----------------------
    Plugin architecture as specified in
    docs/DEB_PACKAGE_ROADMAP.md §4.3.2 "Custom Module Interface"
    """
    
    # Subclasses must define this
    info: PluginInfo = None
    
    # Parameter definitions for GUI generation
    # Each key maps to a dict with: type, default, min, max, description, choices
    parameters: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        """Initialize the plugin."""
        if self.info is None:
            raise ValueError("Plugin must define 'info' class attribute")
    
    @abstractmethod
    def run(self, context: PluginContext, params: Dict[str, Any]) -> PluginResult:
        """
        Execute the plugin.
        
        Parameters
        ----------
        context : PluginContext
            Execution context
        params : Dict[str, Any]
            User-provided parameters
            
        Returns
        -------
        PluginResult
            Execution result
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against definitions.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        
        for name, definition in self.parameters.items():
            value = params.get(name, definition.get("default"))
            
            if value is None and definition.get("required", False):
                errors.append(f"Required parameter '{name}' is missing")
                continue
            
            if value is None:
                continue
            
            param_type = definition.get("type", "str")
            
            # Type validation
            if param_type == "int":
                if not isinstance(value, int):
                    errors.append(f"Parameter '{name}' must be an integer")
                    continue
            elif param_type == "float":
                if not isinstance(value, (int, float)):
                    errors.append(f"Parameter '{name}' must be a number")
                    continue
            elif param_type == "bool":
                if not isinstance(value, bool):
                    errors.append(f"Parameter '{name}' must be a boolean")
                    continue
            elif param_type == "str":
                if not isinstance(value, str):
                    errors.append(f"Parameter '{name}' must be a string")
                    continue
            elif param_type == "choice":
                choices = definition.get("choices", [])
                if value not in choices:
                    errors.append(f"Parameter '{name}' must be one of: {choices}")
                    continue
            
            # Range validation for numbers
            if param_type in ("int", "float"):
                min_val = definition.get("min")
                max_val = definition.get("max")
                
                if min_val is not None and value < min_val:
                    errors.append(f"Parameter '{name}' must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"Parameter '{name}' must be <= {max_val}")
        
        return errors
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameter values.
        
        Returns
        -------
        Dict[str, Any]
            Default parameters
        """
        return {
            name: definition.get("default")
            for name, definition in self.parameters.items()
        }
    
    def get_info(self) -> PluginInfo:
        """Get plugin info."""
        return self.info
    
    def get_name(self) -> str:
        """Get plugin name."""
        return self.info.name
    
    def get_version(self) -> str:
        """Get plugin version."""
        return self.info.version
    
    def get_description(self) -> str:
        """Get plugin description."""
        return self.info.description
    
    def setup(self, context: PluginContext) -> bool:
        """
        Optional setup hook called before first run.
        
        Parameters
        ----------
        context : PluginContext
            Execution context
            
        Returns
        -------
        bool
            True if setup succeeded
        """
        return True
    
    def teardown(self, context: PluginContext) -> None:
        """
        Optional teardown hook called when plugin is unloaded.
        
        Parameters
        ----------
        context : PluginContext
            Execution context
        """
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.info.name} v{self.info.version}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"IRHPlugin({self.info.name!r}, version={self.info.version!r})"


# Registry of plugin classes
_plugin_registry: Dict[str, Type[IRHPlugin]] = {}


def register_plugin(cls: Type[IRHPlugin]) -> Type[IRHPlugin]:
    """
    Decorator to register a plugin class.
    
    Parameters
    ----------
    cls : Type[IRHPlugin]
        Plugin class to register
        
    Returns
    -------
    Type[IRHPlugin]
        The same class (for use as decorator)
        
    Examples
    --------
    >>> @register_plugin
    ... class MyPlugin(IRHPlugin):
    ...     info = PluginInfo(name="My Plugin", ...)
    """
    if cls.info is None:
        raise ValueError(f"Plugin {cls.__name__} must define 'info' attribute")
    
    name = cls.info.name
    _plugin_registry[name] = cls
    logger.info(f"Registered plugin: {name}")
    
    return cls


def get_registered_plugins() -> Dict[str, Type[IRHPlugin]]:
    """
    Get all registered plugin classes.
    
    Returns
    -------
    Dict[str, Type[IRHPlugin]]
        Map of plugin names to classes
    """
    return dict(_plugin_registry)


def unregister_plugin(name: str) -> bool:
    """
    Unregister a plugin.
    
    Parameters
    ----------
    name : str
        Plugin name
        
    Returns
    -------
    bool
        True if unregistered
    """
    if name in _plugin_registry:
        del _plugin_registry[name]
        return True
    return False
